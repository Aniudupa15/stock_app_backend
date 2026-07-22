import uuid
from datetime import datetime

from pydantic import BaseModel


class NotificationOut(BaseModel):
    id: uuid.UUID
    alert_id: uuid.UUID | None
    title: str
    message: str
    created_at: datetime
    read_at: datetime | None
