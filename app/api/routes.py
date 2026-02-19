from fastapi import APIRouter, UploadFile, File, Request, Depends
from sqlmodel import Session
from app.db.session import get_session
import app.services.service as service
import app.services.storage as storage
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
