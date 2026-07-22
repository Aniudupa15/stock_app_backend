from datetime import date
from decimal import Decimal

from pydantic import BaseModel


class CorporateActionOut(BaseModel):
    purpose: str
    face_value: Decimal | None
    ex_date: date | None
    record_date: date | None
    book_closure_start: date | None
    book_closure_end: date | None
