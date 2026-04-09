from fastapi import APIRouter, UploadFile, File, Request, Depends, Query
from sqlmodel import Session
from app.db.session import get_session
import app.services.service as service
import app.services.storage as storage
from app.schemas.dataset import DatasetListResponse
from app.services.dataset import list_dataset
from app.services.train import train_from_db

router = APIRouter()


@router.post("/classify")
async def classify(request: Request, file: UploadFile = File(...)):
    path = await storage.save_upload(file)
    return await service.classify_image(
        path,
        model=request.app.state.model,
        labels=request.app.state.labels,
    )


@router.post("/train")
def train(session: Session = Depends(get_session)):
    return train_from_db(session)


@router.get("/dataset", response_model=DatasetListResponse)
def get_dataset(
    request: Request,
    session: Session = Depends(get_session),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
    human_label: str | None = Query(default=None),
):
    return list_dataset(
        session,
        uploads_base_url=str(request.base_url).rstrip("/") + "/uploads",
        limit=limit,
        offset=offset,
        status=status,
        human_label=human_label,
    )
