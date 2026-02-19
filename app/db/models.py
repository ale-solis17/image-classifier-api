from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional

class Image(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_path: str
    original_name: Optional[str] = None

    predicted_label: Optional[str] = None
    confidence: Optional[float] = None

    status: str = "pending"  # pending | labeled | rejected
    human_label: Optional[str] = None  # SOLO esto se usa para entrenar

    created_at: datetime = Field(default_factory=datetime.utcnow)
