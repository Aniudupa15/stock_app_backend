import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field, model_validator

from app.domain.entities import AlertStatus, AlertType

# Required condition keys per alert type, validated at creation time so the
# evaluation engine never has to handle a malformed condition dict.
_REQUIRED_CONDITION_KEYS: dict[AlertType, set[str]] = {
    AlertType.PRICE_ABOVE: {"price"},
    AlertType.PRICE_BELOW: {"price"},
    AlertType.PERCENT_CHANGE_ABOVE: {"percent"},
    AlertType.PERCENT_CHANGE_BELOW: {"percent"},
    AlertType.RSI_ABOVE: {"threshold"},
    AlertType.RSI_BELOW: {"threshold"},
    AlertType.VOLUME_SPIKE: {"multiplier"},
    AlertType.NEW_52_WEEK_HIGH: set(),
    AlertType.NEW_52_WEEK_LOW: set(),
}


class AlertCreate(BaseModel):
    symbol: str = Field(..., min_length=1)
    alert_type: AlertType
    condition: dict[str, Decimal] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_condition_shape(self) -> "AlertCreate":
        required = _REQUIRED_CONDITION_KEYS[self.alert_type]
        missing = required - self.condition.keys()
        if missing:
            raise ValueError(f"condition for {self.alert_type.value} requires keys: {sorted(required)}")
        return self


class AlertOut(BaseModel):
    id: uuid.UUID
    symbol: str
    alert_type: AlertType
    condition: dict[str, Decimal]
    status: AlertStatus
    created_at: datetime
    triggered_at: datetime | None
