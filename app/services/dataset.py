from pathlib import Path

from sqlmodel import Session, func, select

from app.db.models import Image
from app.schemas.dataset import DatasetImageItem, DatasetListResponse


def list_dataset(
    session: Session,
    *,
    uploads_base_url: str,
    limit: int,
    offset: int,
    status: str | None = None,
    human_label: str | None = None,
) -> DatasetListResponse:
    items_query = select(Image)
    total_query = select(func.count()).select_from(Image)

    if status:
        items_query = items_query.where(Image.status == status)
        total_query = total_query.where(Image.status == status)

    if human_label:
        items_query = items_query.where(Image.human_label == human_label)
        total_query = total_query.where(Image.human_label == human_label)

    rows = session.exec(
        items_query.order_by(Image.created_at.desc()).offset(offset).limit(limit)
    ).all()
    total = session.exec(total_query).one()

    return DatasetListResponse(
        total=int(total),
        limit=limit,
        offset=offset,
        items=[
            DatasetImageItem.model_validate(
                {
                    "id": row.id,
                    "file_path": row.file_path,
                    "image_url": f"{uploads_base_url.rstrip('/')}/{Path(row.file_path).name}",
                    "original_name": row.original_name,
                    "predicted_label": row.predicted_label,
                    "confidence": row.confidence,
                    "status": row.status,
                    "human_label": row.human_label,
                    "created_at": row.created_at,
                }
            )
            for row in rows
        ],
    )
