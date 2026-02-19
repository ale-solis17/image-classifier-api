from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np


def load_image(path: str):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    return arr


def normalize_image(img: Image.Image) -> Image.Image:
    """
    Convierte TIFF de cualquier formato a RGB 8 bits.
    """
    # TIFF/imagen de 16 bits
    if img.mode in ("I;16", "I;16B", "I;16L"):
        arr = np.array(img).astype(np.float32)
        arr = 255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr)

    # Grises/otros → RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def normalize_image_inplace(path: Path, jpg_quality: int = 90) -> None:
    """
    Normaliza la imagen y la reescribe SOBRE el mismo archivo (misma ruta).
    Mantiene el formato según la extensión, con RGB 8 bits.
    """
    ext = path.suffix.lower()

    with Image.open(path) as img:
        # Por si llega un TIFF multi-página: usamos el primer frame
        try:
            img = ImageSequence.Iterator(img).__next__()
        except Exception:
            pass

        img = normalize_image(img)

        if ext in (".tif", ".tiff"):
            # Guardar TIFF normalizado (RGB 8 bits)
            img.save(path, format="TIFF", compression="tiff_deflate")
        elif ext in (".jpg", ".jpeg"):
            img.save(path, format="JPEG", quality=jpg_quality, optimize=True)
        elif ext == ".png":
            img.save(path, format="PNG", optimize=True)
        else:
            # Si llega algo raro, al menos lo dejás en RGB re-guardando
            img.save(path)
