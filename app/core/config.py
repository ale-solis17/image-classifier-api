import os
from pathlib import Path
from dotenv import load_dotenv

# Root del proyecto (…/classifier-api)
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

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
UNKNOWN_LABEL = "No reconocido"
UNKNOWN_THRESHOLD = 0.70
MIN_CONFIDENCE_MARGIN = 0.10

# OpenAI chat
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
OPENAI_CHAT_TIMEOUT_S = float(os.getenv("OPENAI_CHAT_TIMEOUT_S", "30"))
OPENAI_CHAT_MAX_MESSAGES = int(os.getenv("OPENAI_CHAT_MAX_MESSAGES", "8"))
OPENAI_CHAT_MAX_CHARS = int(os.getenv("OPENAI_CHAT_MAX_CHARS", "4000"))
