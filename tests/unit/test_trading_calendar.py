"""Unit tests for libs/trading_calendar. Pure - no DB, no Docker.

Reference dates (verified weekdays):
  2026-01-05 Monday, 2026-01-06 Tuesday, 2026-01-02 Friday,
  2026-01-03 Saturday, 2026-01-04 Sunday, 2026-11-08 Sunday (Muhurat sample).
"""

from datetime import date, datetime, time

from libs.trading_calendar.calendar import MarketState, TradingCalendar

MON = date(2026, 1, 5)
TUE = date(2026, 1, 6)
FRI = date(2026, 1, 2)
SAT = date(2026, 1, 3)


def _cal(holidays=None, special=None):
    return TradingCalendar(holidays=holidays, special_sessions=special)


def test_weekend_is_not_trading_day():
    cal = _cal()
    assert cal.is_trading_day(MON) is True
    assert cal.is_trading_day(SAT) is False


def test_injected_holiday_is_not_trading_day():
    cal = _cal(holidays={TUE})
    assert cal.is_trading_day(TUE) is False
    assert cal.is_trading_day(MON) is True


def test_next_and_previous_trading_day_skip_weekend_and_holidays():
    cal = _cal(holidays={MON})  # Mon holiday -> Fri <-> Tue across the weekend+holiday
    assert cal.next_trading_day(FRI) == TUE
    assert cal.previous_trading_day(TUE) == FRI


def test_state_transitions_through_the_day():
    cal = _cal()
    assert cal.state_at(datetime.combine(MON, time(8, 30))) is MarketState.CLOSED
    assert cal.state_at(datetime.combine(MON, time(9, 5))) is MarketState.PRE_OPEN
    assert cal.state_at(datetime.combine(MON, time(9, 20))) is MarketState.CONTINUOUS
    assert cal.state_at(datetime.combine(MON, time(15, 25))) is MarketState.CONTINUOUS
    assert cal.state_at(datetime.combine(MON, time(15, 35))) is MarketState.POST_CLOSE
    assert cal.state_at(datetime.combine(MON, time(16, 30))) is MarketState.CLOSED


def test_closed_all_day_on_non_trading_day():
    cal = _cal()
    assert cal.state_at(datetime.combine(SAT, time(11, 0))) is MarketState.CLOSED


def test_is_open_only_during_continuous():
    cal = _cal()
    assert cal.is_open(datetime.combine(MON, time(9, 20))) is True
    assert cal.is_open(datetime.combine(MON, time(9, 5))) is False
    assert cal.is_open(datetime.combine(MON, time(15, 35))) is False


def test_square_off_window_default_buffer():
    cal = _cal()
    assert cal.square_off_start(MON) == time(15, 15)
    assert cal.in_square_off_window(datetime.combine(MON, time(15, 10))) is False
    assert cal.in_square_off_window(datetime.combine(MON, time(15, 20))) is True
    assert cal.in_square_off_window(datetime.combine(MON, time(15, 29))) is True
    # end is exclusive - at 15:30 continuous is over, square-off window closed
    assert cal.in_square_off_window(datetime.combine(MON, time(15, 30))) is False


def test_square_off_custom_buffer():
    cal = _cal()
    assert cal.square_off_start(MON, buffer_minutes=30) == time(15, 0)
    assert cal.in_square_off_window(datetime.combine(MON, time(15, 5)), buffer_minutes=30) is True


def test_seconds_until_close():
    cal = _cal()
    assert cal.seconds_until_close(datetime.combine(MON, time(15, 0))) == 1800
    # not in continuous session -> nothing to count down to
    assert cal.seconds_until_close(datetime.combine(MON, time(8, 0))) is None


def test_special_muhurat_session_on_a_sunday():
    muhurat = date(2026, 11, 8)  # Sunday
    cal = _cal(special={muhurat: (time(18, 15), time(19, 15))})
    assert cal.is_trading_day(muhurat) is True
    assert cal.is_open(datetime.combine(muhurat, time(18, 30))) is True
    assert cal.is_open(datetime.combine(muhurat, time(10, 0))) is False
    assert cal.square_off_start(muhurat) == time(19, 0)
