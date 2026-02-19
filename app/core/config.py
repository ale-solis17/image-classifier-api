from pathlib import Path

# Root del proyecto (…/classifier-api)
BASE_DIR = Path(__file__).resolve().parents[2]

# Datos y DB
DATA_DIR = (BASE_DIR / "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = (DATA_DIR / "database.db").resolve()
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

# Images
UPLOAD_DIR = (DATA_DIR / "uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

# Model files
MODELS_DIR = (BASE_DIR / "models")
MODEL_PATH = (MODELS_DIR / "model.keras").resolve()
LABELS_PATH = (MODELS_DIR / "labels.json").resolve()

# Params
UNKNOWN_THRESHOLD = 0.60
