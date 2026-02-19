from pathlib import Path
import shutil
import uuid

from sqlmodel import Session

from app.core.config import UPLOAD_DIR, DATABASE_URL
from app.db.session import engine, create_db_and_tables
from app.db.models import Image
from app.services.preprocess import normalize_image_inplace

# Carpeta donde tienes tu dataset inicial (por clases)
# Ej:
# seed_dataset/
#   Bacteroides fragilis/
#     img1.jpg
#   Staphylococcus aureus/
#     img2.jpg
SEED_DIR = Path("seed_dataset")

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def main():
    print("DB:", DATABASE_URL)
    print("SEED_DIR:", SEED_DIR.resolve())
    print("UPLOAD_DIR:", UPLOAD_DIR.resolve())

    if not SEED_DIR.exists():
        raise FileNotFoundError(f"No existe {SEED_DIR}. Crea esa carpeta y pon clases dentro.")

    create_db_and_tables()

    inserted = 0
    with Session(engine) as session:
        for class_dir in sorted(SEED_DIR.iterdir()):
            if not class_dir.is_dir():
                continue

            label = class_dir.name.strip()
            files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT]

            print(f"Clase '{label}': {len(files)} archivos")

            for src in files:
                new_name = f"{uuid.uuid4()}{src.suffix.lower()}"
                dst = (UPLOAD_DIR / new_name)

                shutil.copy2(src, dst)

                # Normaliza TIFF/16-bit/multipage, etc (tu función)
                normalize_image_inplace(dst)

                row = Image(
                    file_path=str(dst.resolve()),
                    original_name=src.name,
                    human_label=label,
                    status="labeled",
                )
                session.add(row)
                inserted += 1

        session.commit()

    print(f"✅ Insertados en DB: {inserted}")
    if inserted == 0:
        print("⚠️ No se importó nada. Revisa que seed_dataset tenga subcarpetas con imágenes válidas.")


if __name__ == "__main__":
    main()
