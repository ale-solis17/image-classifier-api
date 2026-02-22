from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path

from app.api.routes import router
from app.db.session import create_db_and_tables
from app.ai.loader import load_labels, load_model
from app.core.config import MODEL_PATH

from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()

    app.state.model = None
    app.state.labels = None

    if Path(MODEL_PATH).exists():
        app.state.model = load_model()
        app.state.labels = load_labels()
        print("✅ Modelo cargado en startup")
    else:
        print("⚠️ No hay model.keras todavía. Levanto la API sin modelo. Corre POST /train.")

    yield


app = FastAPI(title="Image Classifier API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)