from app.core.config import MODEL_PATH, LABELS_PATH
from tensorflow.keras.models import load_model as keras_load_model
from functools import lru_cache
import json

@lru_cache(maxsize=1)
def load_model():
    """Carga el modelo entrenado desde disco (cacheado)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo en: {MODEL_PATH}")

    # Tip: en Keras suele acelerar/des complicar la carga
    return keras_load_model(MODEL_PATH, compile=False)

@lru_cache(maxsize=1)
def load_labels():
    """Carga labels desde disco (cacheado)."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"No existen labels en: {LABELS_PATH}")

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError("labels.json debe ser una lista de strings")

    # opcional: limpiar espacios
    labels = [x.strip() for x in labels]
    return labels
