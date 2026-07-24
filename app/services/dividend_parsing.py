import re
from decimal import Decimal

_DIVIDEND_AMOUNT_PATTERN = re.compile(r"R[se]\.?\s*([\d,]+(?:\.\d+)?)\s*Per Share", re.IGNORECASE)


def sum_dividend_amount(purpose: str) -> Decimal:
    """Corporate-action `purpose` is free text (e.g. "Dividend - Rs 10 Per
    Share/Special Dividend - Rs 30 Per Share") - sums every "Rs/Re X Per
    Share" amount found, since one action can bundle multiple payouts.
    Shared between FundamentalsService (per-symbol yield) and DividendService
    (cross-stock dividend list) so the parsing rule lives in exactly one place.
    """
    total = Decimal("0")
    for match in _DIVIDEND_AMOUNT_PATTERN.finditer(purpose):
        try:
            total += Decimal(match.group(1).replace(",", ""))
        except Exception:
            continue
    return total
