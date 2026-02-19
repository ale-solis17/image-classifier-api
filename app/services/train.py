from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sqlmodel import Session, select

from app.core.config import MODELS_DIR, LABELS_PATH, MODEL_PATH
from app.db.models import Image
from app.services.preprocess import normalize_image

# ===== Config =====
IMG_SIZE = (224, 224)
BATCH = 16
EPOCHS = 5
VAL_SPLIT = 0.2
SEED = 1337


@dataclass
class TrainResult:
    trained_on: int
    classes: List[str]
    metrics: dict
    model_path: str
    labels_path: str


def _fetch_labeled_samples(session: Session) -> List[Tuple[str, str]]:
    """Devuelve [(file_path, human_label), ...] solo de imágenes etiquetadas."""
    rows = session.exec(
        select(Image)
        .where(Image.status == "labeled")
        .where(Image.human_label != None)  # noqa: E711
    ).all()

    samples: List[Tuple[str, str]] = []
    for r in rows:
        if not r.file_path or not r.human_label:
            continue
        p = Path(r.file_path)
        if p.exists():
            samples.append((str(p), r.human_label))
    return samples


def _build_label_map(samples: List[Tuple[str, str]]) -> Tuple[List[str], dict]:
    labels = sorted({lbl for _, lbl in samples})
    return labels, {lbl: i for i, lbl in enumerate(labels)}


def _split(paths: List[str], y: List[int]) -> Tuple[List[str], List[int], List[str], List[int]]:
    rng = random.Random(SEED)
    idx = list(range(len(paths)))
    rng.shuffle(idx)

    n_val = max(1, int(len(idx) * VAL_SPLIT))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_paths = [paths[i] for i in train_idx]
    train_y = [y[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_y = [y[i] for i in val_idx]
    return train_paths, train_y, val_paths, val_y


def _pil_load(path_tensor) -> np.ndarray:
    """
    path_tensor llega como tf.EagerTensor(dtype=string). Hay que convertirlo.
    Devuelve float32 0..255 (SIN /255) para usar preprocess_input.
    """
    from PIL import Image as PILImage

    # convertir Tensor -> str
    path = path_tensor.numpy().decode("utf-8")

    with PILImage.open(path) as img:
        img = normalize_image(img)      # tu función (RGB 8-bit)
        img = img.resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32)  # 0..255
    return arr


def _make_dataset(paths: List[str], y: List[int], training: bool) -> tf.data.Dataset:
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    y_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y, tf.int32))
    ds = tf.data.Dataset.zip((path_ds, y_ds))

    def _load(path: tf.Tensor, label: tf.Tensor):
        img = tf.py_function(func=_pil_load, inp=[path], Tout=tf.float32)
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds


def train_from_db(session: Session) -> TrainResult:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    samples = _fetch_labeled_samples(session)
    if len(samples) < 10:
        raise ValueError(f"Necesitas más data etiquetada para entrenar. Encontré {len(samples)} muestras labeled.")

    labels, label_to_idx = _build_label_map(samples)
    if len(labels) < 2:
        raise ValueError("Para entrenar necesitas al menos 2 clases distintas en human_label.")

    # Guardar labels.json en el orden exacto del modelo
    LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    paths = [p for p, _ in samples]
    y = [label_to_idx[lbl] for _, lbl in samples]
    train_paths, train_y, val_paths, val_y = _split(paths, y)

    train_ds = _make_dataset(train_paths, train_y, training=True)
    val_ds = _make_dataset(val_paths, val_y, training=False)

    # Base preentrenado
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)  # espera 0..255 float
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(labels), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

    # Guardar modelo real
    model.save(MODEL_PATH)

    last = {k: float(v[-1]) for k, v in hist.history.items()}
    return TrainResult(
        trained_on=len(samples),
        classes=labels,
        metrics=last,
        model_path=str(MODEL_PATH),
        labels_path=str(LABELS_PATH),
    )
