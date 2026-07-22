from datetime import date
from decimal import Decimal

from app.providers.nse.xbrl import parse_xbrl_document, select_quarterly_context

# A minimal but structurally real in-bse-fin XBRL instance: two duration
# contexts ending on the SAME date (2024-12-31) - one a genuine quarter
# (~92 days), one a 9-month cumulative span (~276 days) - a scenario-tagged
# (segment) context that must be excluded, and an instant (balance-sheet-style)
# context that must also be excluded from quarter selection.
_SAMPLE_XBRL = b"""<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
            xmlns:in-bse-fin="http://www.bseindia.com/xbrl/fin/2020-03-31/in-bse-fin"
            xmlns:xbrldi="http://xbrl.org/2006/xbrldi"
            xmlns:link="http://www.xbrl.org/2003/linkbase"
            xmlns:xlink="http://www.w3.org/1999/xlink">
<link:schemaRef xlink:type="simple" xlink:href="Ind-AS_entry_point_2020-03-31.xsd"/>
<xbrli:context id="QuarterCtx">
  <xbrli:entity><xbrli:identifier scheme="http://www.nseindia.com/NSESymbol">TESTCO</xbrli:identifier></xbrli:entity>
  <xbrli:period><xbrli:startDate>2024-10-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period>
</xbrli:context>
<xbrli:context id="NineMonthCtx">
  <xbrli:entity><xbrli:identifier scheme="http://www.nseindia.com/NSESymbol">TESTCO</xbrli:identifier></xbrli:entity>
  <xbrli:period><xbrli:startDate>2024-04-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period>
</xbrli:context>
<xbrli:context id="SegmentCtx">
  <xbrli:entity><xbrli:identifier scheme="http://www.nseindia.com/NSESymbol">TESTCO</xbrli:identifier></xbrli:entity>
  <xbrli:period><xbrli:startDate>2024-10-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period>
  <xbrli:scenario><xbrldi:explicitMember dimension="in-bse-fin:SegmentAxis">in-bse-fin:SegmentAMember</xbrldi:explicitMember></xbrli:scenario>
</xbrli:context>
<xbrli:context id="InstantCtx">
  <xbrli:entity><xbrli:identifier scheme="http://www.nseindia.com/NSESymbol">TESTCO</xbrli:identifier></xbrli:entity>
  <xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period>
</xbrli:context>
<in-bse-fin:RevenueFromOperations contextRef="QuarterCtx" unitRef="INR" decimals="-3">100000.00</in-bse-fin:RevenueFromOperations>
<in-bse-fin:RevenueFromOperations contextRef="NineMonthCtx" unitRef="INR" decimals="-3">280000.00</in-bse-fin:RevenueFromOperations>
<in-bse-fin:RevenueFromOperations contextRef="SegmentCtx" unitRef="INR" decimals="-3">40000.00</in-bse-fin:RevenueFromOperations>
<in-bse-fin:ProfitLossForPeriod contextRef="QuarterCtx" unitRef="INR" decimals="-3">15000.00</in-bse-fin:ProfitLossForPeriod>
<in-bse-fin:ProfitLossForPeriod contextRef="NineMonthCtx" unitRef="INR" decimals="-3">42000.00</in-bse-fin:ProfitLossForPeriod>
<in-bse-fin:BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations contextRef="QuarterCtx" unitRef="INRPerShare" decimals="INF">1.25</in-bse-fin:BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations>
</xbrli:xbrl>
"""


def test_parse_xbrl_document_extracts_contexts_and_facts():
    doc = parse_xbrl_document(_SAMPLE_XBRL)

    assert set(doc.contexts.keys()) == {"QuarterCtx", "NineMonthCtx", "SegmentCtx", "InstantCtx"}
    assert doc.contexts["QuarterCtx"].start == date(2024, 10, 1)
    assert doc.contexts["QuarterCtx"].end == date(2024, 12, 31)
    assert doc.contexts["QuarterCtx"].has_scenario is False
    assert doc.contexts["SegmentCtx"].has_scenario is True
    assert doc.contexts["InstantCtx"].start == doc.contexts["InstantCtx"].end == date(2024, 12, 31)

    revenue_facts = [f for f in doc.facts if f.concept == "RevenueFromOperations"]
    assert len(revenue_facts) == 3


def test_select_quarterly_context_prefers_true_quarter_over_cumulative():
    """The core regression this whole design exists for: two contexts end on
    the same date, one is a real quarter (~90 days), one is 9-month
    cumulative (~276 days) - selection must go by duration, not by
    whichever context happens to be listed/named first.
    """
    doc = parse_xbrl_document(_SAMPLE_XBRL)

    selected = select_quarterly_context(doc.contexts, period_end=date(2024, 12, 31))

    assert selected == "QuarterCtx"


def test_select_quarterly_context_excludes_segment_and_instant_contexts():
    doc = parse_xbrl_document(_SAMPLE_XBRL)
    selected = select_quarterly_context(doc.contexts, period_end=date(2024, 12, 31))
    assert selected != "SegmentCtx"
    assert selected != "InstantCtx"


def test_select_quarterly_context_returns_none_when_no_context_matches_period_end():
    doc = parse_xbrl_document(_SAMPLE_XBRL)
    selected = select_quarterly_context(doc.contexts, period_end=date(2020, 1, 1))
    assert selected is None


def test_facts_for_selected_context_give_correct_quarterly_values():
    doc = parse_xbrl_document(_SAMPLE_XBRL)
    context_id = select_quarterly_context(doc.contexts, period_end=date(2024, 12, 31))

    values = {f.concept: f.value for f in doc.facts if f.context_id == context_id}

    assert Decimal(values["RevenueFromOperations"]) == Decimal("100000.00")
    assert Decimal(values["ProfitLossForPeriod"]) == Decimal("15000.00")
    assert Decimal(values["BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations"]) == Decimal("1.25")


def test_select_quarterly_context_within_tolerance_for_near_end_dates():
    doc = parse_xbrl_document(_SAMPLE_XBRL)
    # 2 days off the declared period_end - within the default 5-day tolerance
    selected = select_quarterly_context(doc.contexts, period_end=date(2025, 1, 2))
    assert selected == "QuarterCtx"


def test_select_quarterly_context_respects_custom_tolerance():
    doc = parse_xbrl_document(_SAMPLE_XBRL)
    selected = select_quarterly_context(doc.contexts, period_end=date(2025, 1, 2), tolerance_days=1)
    assert selected is None
