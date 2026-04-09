from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DatasetImageItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_path: str
    image_url: str
    original_name: str | None
    predicted_label: str | None
    confidence: float | None
    status: str
    human_label: str | None
    created_at: datetime


class DatasetListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[DatasetImageItem]
