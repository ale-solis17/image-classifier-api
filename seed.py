from pathlib import Path
import shutil
import uuid

from sqlmodel import Session
from app.db.session import engine
from app.db.models import Image
from app.core.config import UPLOAD_DIR
from app.services.preprocess import normalize_image_inplace

SEED_DIR = Path("seed_dataset")


def main():
    with Session(engine) as session:

        for class_dir in SEED_DIR.iterdir():
            if not class_dir.is_dir():
                continue

            label = class_dir.name

            for img in class_dir.iterdir():
                if not img.is_file():
                    continue

                new_name = f"{uuid.uuid4()}{img.suffix.lower()}"
                dst = UPLOAD_DIR / new_name

                shutil.copy(img, dst)

                # normalizar (tu función)
                normalize_image_inplace(dst)

                row = Image(
                    file_path=str(dst),
                    original_name=img.name,
                    human_label=label,
                    predicted_label=None,
                    confidence=None,
                    status="labeled",  # ← importante
                )

                session.add(row)

        session.commit()
        print("Dataset inicial cargado ✔")


if __name__ == "__main__":
    main()
