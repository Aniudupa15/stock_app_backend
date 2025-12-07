"""
Market Schedule Module
Handles NSE market timings, holidays, and market status checks
"""

from datetime import datetime, time, timedelta
from typing import Dict, List, Optional
import pytz

class MarketSchedule:
    """
    NSE (National Stock Exchange of India) Market Schedule Handler
    
    Market Timings:
    - Pre-market: 09:00 - 09:15 IST
    - Market Hours: 09:15 - 15:30 IST
    - Post-market: 15:40 - 16:00 IST
    
    Trading Days: Monday - Friday (excluding holidays)
    """
    
    # NSE Market Timings (IST)
    TIMEZONE = pytz.timezone('Asia/Kolkata')
    
    # Market session times
    PRE_MARKET_START = time(9, 0)
    MARKET_OPEN = time(9, 15)
    MARKET_CLOSE = time(15, 30)
    POST_MARKET_END = time(16, 0)
    
    # NSE Holidays 2024-2025
    HOLIDAYS = {
        # 2024 Holidays
        "2024-01-26": "Republic Day",
        "2024-03-08": "Mahashivratri",
        "2024-03-25": "Holi",
        "2024-03-29": "Good Friday",
        "2024-04-11": "Id-Ul-Fitr",
        "2024-04-17": "Ram Navami",
        "2024-04-21": "Mahavir Jayanti",
        "2024-05-01": "Maharashtra Day",
        "2024-05-23": "Buddha Pournima",
        "2024-06-17": "Bakri Id",
        "2024-07-17": "Muharram",
        "2024-08-15": "Independence Day",
        "2024-08-26": "Ganesh Chaturthi",
        "2024-10-02": "Mahatma Gandhi Jayanti",
        "2024-10-12": "Dussehra",
        "2024-11-01": "Diwali Laxmi Pujan",
        "2024-11-02": "Diwali Balipratipada",
        "2024-11-15": "Gurunanak Jayanti",
        "2024-12-25": "Christmas",
        
        # 2025 Holidays (Tentative - subject to change)
        "2025-01-26": "Republic Day",
        "2025-02-26": "Mahashivratri",
        "2025-03-14": "Holi",
        "2025-03-31": "Id-Ul-Fitr",
        "2025-04-06": "Ram Navami",
        "2025-04-10": "Mahavir Jayanti",
        "2025-04-18": "Good Friday",
        "2025-05-01": "Maharashtra Day",
        "2025-05-12": "Buddha Pournima",
        "2025-06-07": "Bakri Id",
        "2025-07-05": "Muharram",
        "2025-08-15": "Independence Day",
        "2025-08-27": "Ganesh Chaturthi",
        "2025-10-02": "Mahatma Gandhi Jayanti",
        "2025-10-21": "Dussehra",
        "2025-10-20": "Diwali Laxmi Pujan",
        "2025-11-05": "Gurunanak Jayanti",
        "2025-12-25": "Christmas"
    }
    
    def __init__(self):
        """Initialize MarketSchedule"""
        pass
    
    def get_current_time_ist(self) -> datetime:
        """Get current time in IST"""
        return datetime.now(self.TIMEZONE)
    
    def is_trading_day(self, date: Optional[datetime] = None) -> bool:
        """
        Check if given date is a trading day
        
        Args:
            date: Date to check (defaults to today)
            
        Returns:
            True if it's a trading day, False otherwise
        """
        if date is None:
            date = self.get_current_time_ist()
        
        # Check if weekend
        if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        
        # Check if holiday
        date_str = date.strftime("%Y-%m-%d")
        if date_str in self.HOLIDAYS:
            return False
        
        return True
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open
        
        Returns:
            True if market is open, False otherwise
        """
        now = self.get_current_time_ist()
        
        # Check if trading day
        if not self.is_trading_day(now):
            return False
        
        # Check if within market hours
        current_time = now.time()
        return self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE
    
    def get_market_status(self) -> Dict:
        """
        Get detailed market status
        
        Returns:
            Dictionary with market status information
        """
        now = self.get_current_time_ist()
        current_time = now.time()
        
        is_trading_day = self.is_trading_day(now)
        is_open = False
        session = "CLOSED"
        
        if is_trading_day:
            if current_time < self.PRE_MARKET_START:
                session = "PRE_OPEN"
            elif self.PRE_MARKET_START <= current_time < self.MARKET_OPEN:
                session = "PRE_MARKET"
            elif self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE:
                session = "MARKET_HOURS"
                is_open = True
            elif self.MARKET_CLOSE < current_time <= self.POST_MARKET_END:
                session = "POST_MARKET"
            else:
                session = "CLOSED"
        else:
            if now.weekday() >= 5:
                session = "WEEKEND"
            else:
                date_str = now.strftime("%Y-%m-%d")
                if date_str in self.HOLIDAYS:
                    session = f"HOLIDAY - {self.HOLIDAYS[date_str]}"
        
        return {
            "is_open": is_open,
            "session": session,
            "is_trading_day": is_trading_day,
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "day_of_week": now.strftime("%A")
        }
    
    def get_market_timings(self) -> Dict:
        """
        Get market timings
        
        Returns:
            Dictionary with market session timings
        """
        return {
            "timezone": "Asia/Kolkata (IST)",
            "pre_market": {
                "start": self.PRE_MARKET_START.strftime("%H:%M"),
                "end": self.MARKET_OPEN.strftime("%H:%M")
            },
            "market_hours": {
                "open": self.MARKET_OPEN.strftime("%H:%M"),
                "close": self.MARKET_CLOSE.strftime("%H:%M")
            },
            "post_market": {
                "start": "15:40",
                "end": self.POST_MARKET_END.strftime("%H:%M")
            },
            "trading_days": "Monday - Friday (excluding holidays)"
        }
    
    def time_until_market_open(self) -> Dict:
        """
        Calculate time remaining until market opens
        
        Returns:
            Dictionary with countdown information
        """
        now = self.get_current_time_ist()
        
        if self.is_market_open():
            return {
                "status": "MARKET_OPEN",
                "message": "Market is currently open"
            }
        
        # Find next market open
        next_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        
        # If today's market has closed, move to next trading day
        if now.time() > self.MARKET_CLOSE:
            next_open += timedelta(days=1)
        
        # Skip to next trading day
        while not self.is_trading_day(next_open):
            next_open += timedelta(days=1)
        
        time_diff = next_open - now
        hours = int(time_diff.total_seconds() // 3600)
        minutes = int((time_diff.total_seconds() % 3600) // 60)
        
        return {
            "status": "MARKET_CLOSED",
            "next_open": next_open.strftime("%Y-%m-%d %H:%M:%S"),
            "hours_until_open": hours,
            "minutes_until_open": minutes,
            "message": f"Market opens in {hours}h {minutes}m"
        }
    
    def time_until_market_close(self) -> Dict:
        """
        Calculate time remaining until market closes
        
        Returns:
            Dictionary with countdown information
        """
        now = self.get_current_time_ist()
        
        if not self.is_market_open():
            return {
                "status": "MARKET_CLOSED",
                "message": "Market is currently closed"
            }
        
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        time_diff = market_close - now
        
        hours = int(time_diff.total_seconds() // 3600)
        minutes = int((time_diff.total_seconds() % 3600) // 60)
        
        return {
            "status": "MARKET_OPEN",
            "closes_at": market_close.strftime("%Y-%m-%d %H:%M:%S"),
            "hours_until_close": hours,
            "minutes_until_close": minutes,
            "message": f"Market closes in {hours}h {minutes}m"
        }
    
    def get_upcoming_holidays(self, days: int = 90) -> List[Dict]:
        """
        Get upcoming market holidays
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            List of upcoming holidays
        """
        now = self.get_current_time_ist()
        end_date = now + timedelta(days=days)
        
        upcoming = []
        for date_str, holiday_name in self.HOLIDAYS.items():
            holiday_date = datetime.strptime(date_str, "%Y-%m-%d")
            holiday_date = self.TIMEZONE.localize(holiday_date)
            
            if now <= holiday_date <= end_date:
                days_away = (holiday_date.date() - now.date()).days
                upcoming.append({
                    "date": date_str,
                    "holiday": holiday_name,
                    "day_of_week": holiday_date.strftime("%A"),
                    "days_away": days_away
                })
        
        # Sort by date
        upcoming.sort(key=lambda x: x['date'])
        
        return upcoming
    
    def get_next_trading_day(self, date: Optional[datetime] = None) -> datetime:
        """
        Get next trading day after given date
        
        Args:
            date: Starting date (defaults to today)
            
        Returns:
            Next trading day
        """
        if date is None:
            date = self.get_current_time_ist()
        
        next_day = date + timedelta(days=1)
        
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        
        return next_day
    
    def get_trading_days_in_range(self, start_date: datetime, end_date: datetime) -> int:
        """
        Count trading days in a date range
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Number of trading days
        """
        count = 0
        current = start_date
        
        while current <= end_date:
            if self.is_trading_day(current):
                count += 1
            current += timedelta(days=1)
        
        return count