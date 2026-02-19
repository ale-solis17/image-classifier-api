from fastapi import UploadFile, HTTPException
from pathlib import Path
import uuid
from app.core.config import UPLOAD_DIR, VALID_EXT
from app.services.preprocess import normalize_image_inplace


async def save_upload(file: UploadFile) -> str:
    suffix = valid_suffix(Path(file.filename))
    if suffix == "":
        raise HTTPException(status_code=400, detail=f"Archivo no soportado. Extensiones válidas: {VALID_EXT}")

    ext = Path(file.filename).suffix.lower()
    filename = f"{uuid.uuid4()}{ext}"
    path = UPLOAD_DIR / filename

    contents = await file.read()
    path.write_bytes(contents)

    normalize_image_inplace(path)

    return str(path)


def valid_suffix(path: Path) -> str:
    response:str = ""
    if path.suffix.lower() in VALID_EXT:
        response = path.suffix.lower()
    return response