"""Parses NSE/BSE XBRL financial-results filings (`in-bse-fin` taxonomy) into
a generic (contexts, facts) structure.

Concept facts are matched by local tag name only, ignoring the namespace URI -
the `in-bse-fin` taxonomy is versioned by date in its namespace URI
(e.g. `.../2020-03-31/in-bse-fin`) and that version can vary by filing year,
but the concept names themselves (`RevenueFromOperations`, etc.) are stable
within the "Ind-AS New" taxonomy family.

XBRL context IDs (e.g. "OneD", "FourD") are arbitrary per-filer strings, NOT
standardized - `select_quarterly_context` below matches by each context's own
declared period dates instead of guessing at ID conventions.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime

_XBRLI_NS = "{http://www.xbrl.org/2003/instance}"
_TARGET_QUARTER_DAYS = 90


@dataclass(frozen=True, slots=True)
class XbrlContext:
    start: date | None
    end: date | None
    has_scenario: bool


@dataclass(frozen=True, slots=True)
class XbrlFact:
    concept: str
    context_id: str
    value: str


@dataclass(frozen=True, slots=True)
class XbrlDocument:
    contexts: dict[str, XbrlContext]
    facts: list[XbrlFact]


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _parse_xbrl_date(text: str) -> date | None:
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_xbrl_document(content: bytes) -> XbrlDocument:
    root = ET.fromstring(content)

    contexts: dict[str, XbrlContext] = {}
    for ctx_el in root.findall(f"{_XBRLI_NS}context"):
        context_id = ctx_el.get("id")
        if not context_id:
            continue
        period_el = ctx_el.find(f"{_XBRLI_NS}period")
        start = end = None
        if period_el is not None:
            instant_el = period_el.find(f"{_XBRLI_NS}instant")
            if instant_el is not None and instant_el.text:
                start = end = _parse_xbrl_date(instant_el.text)
            else:
                start_el = period_el.find(f"{_XBRLI_NS}startDate")
                end_el = period_el.find(f"{_XBRLI_NS}endDate")
                if start_el is not None and start_el.text:
                    start = _parse_xbrl_date(start_el.text)
                if end_el is not None and end_el.text:
                    end = _parse_xbrl_date(end_el.text)
        has_scenario = ctx_el.find(f"{_XBRLI_NS}scenario") is not None
        contexts[context_id] = XbrlContext(start=start, end=end, has_scenario=has_scenario)

    facts: list[XbrlFact] = []
    for el in root:
        if el.tag.startswith(_XBRLI_NS) or _local_name(el.tag) == "schemaRef":
            continue
        context_id = el.get("contextRef")
        if not context_id or el.text is None or not el.text.strip():
            continue
        facts.append(XbrlFact(concept=_local_name(el.tag), context_id=context_id, value=el.text.strip()))

    return XbrlDocument(contexts=contexts, facts=facts)


def select_quarterly_context(contexts: dict[str, XbrlContext], period_end: date, tolerance_days: int = 5) -> str | None:
    """Among undimensioned duration contexts ending near `period_end`, pick the
    one whose duration is closest to 90 days (a single quarter - not a 9-month
    or annual cumulative span, which the same filing's other contexts often carry).
    """
    candidates: list[tuple[int, str]] = []
    for context_id, ctx in contexts.items():
        if ctx.has_scenario or ctx.start is None or ctx.end is None or ctx.start == ctx.end:
            continue
        if abs((ctx.end - period_end).days) > tolerance_days:
            continue
        duration = (ctx.end - ctx.start).days
        candidates.append((abs(duration - _TARGET_QUARTER_DAYS), context_id))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]
