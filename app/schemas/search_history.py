from datetime import datetime

from pydantic import BaseModel


class SearchHistoryEntryOut(BaseModel):
    query: str
    searched_at: datetime
