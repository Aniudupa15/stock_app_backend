import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class IPOAnalyzer:
    """
    IPO tracking and analysis system
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.ipos_file = self.data_dir / "ipos.json"
        self.ipo_reminders_file = self.data_dir / "ipo_reminders.json"
        
        self._init_files()
        self._initialize_sample_ipos()
    
    def _init_files(self):
        """Initialize JSON files"""
        if not self.ipos_file.exists():
            self._save_json(self.ipos_file, [])
        
        if not self.ipo_reminders_file.exists():
            self._save_json(self.ipo_reminders_file, [])
    
    def _load_json(self, filepath: Path) -> any:
        """Load JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def _save_json(self, filepath: Path, data: any):
        """Save to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving {filepath}: {e}")
    
    def _initialize_sample_ipos(self):
        """Initialize with sample upcoming IPOs"""
        ipos = self._load_json(self.ipos_file)
        
        if not ipos:  # Only initialize if empty
            today = datetime.now()
            
            sample_ipos = [
                {
                    "id": 1,
                    "company_name": "Sample Tech IPO",
                    "open_date": (today + timedelta(days=7)).strftime("%Y-%m-%d"),
                    "close_date": (today + timedelta(days=10)).strftime("%Y-%m-%d"),
                    "listing_date": (today + timedelta(days=15)).strftime("%Y-%m-%d"),
                    "price_band": "₹300-350",
                    "lot_size": 40,
                    "issue_size": "₹500 Cr",
                    "category": "Mainboard",
                    "status": "Upcoming"
                }
            ]
            
            self._save_json(self.ipos_file, sample_ipos)
    
    def add_ipo(
        self,
        company_name: str,
        open_date: str,
        close_date: str,
        listing_date: str,
        price_band: str,
        lot_size: int,
        issue_size: str,
        category: str = "Mainboard",
        sector: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Add a new IPO for tracking
        
        Args:
            company_name: Name of the company
            open_date: IPO open date (YYYY-MM-DD)
            close_date: IPO close date (YYYY-MM-DD)
            listing_date: Expected listing date (YYYY-MM-DD)
            price_band: Price band (e.g., "₹100-120")
            lot_size: Minimum lot size
            issue_size: Total issue size (e.g., "₹500 Cr")
            category: Mainboard or SME
            sector: Industry sector
            notes: Additional notes
            
        Returns:
            Created IPO details
        """
        ipos = self._load_json(self.ipos_file)
        
        ipo = {
            "id": len(ipos) + 1,
            "company_name": company_name,
            "open_date": open_date,
            "close_date": close_date,
            "listing_date": listing_date,
            "price_band": price_band,
            "lot_size": lot_size,
            "issue_size": issue_size,
            "category": category,
            "sector": sector,
            "status": self._determine_status(open_date, close_date),
            "notes": notes,
            "added_at": datetime.now().isoformat(),
            "analysis": None
        }
        
        ipos.append(ipo)
        self._save_json(self.ipos_file, ipos)
        
        return {
            "success": True,
            "message": f"IPO added: {company_name}",
            "ipo": ipo
        }
    
    def _determine_status(self, open_date: str, close_date: str) -> str:
        """Determine IPO status based on dates"""
        today = datetime.now().date()
        open_dt = datetime.strptime(open_date, "%Y-%m-%d").date()
        close_dt = datetime.strptime(close_date, "%Y-%m-%d").date()
        
        if today < open_dt:
            return "Upcoming"
        elif open_dt <= today <= close_dt:
            return "Open"
        else:
            return "Closed"
    
    def get_ipos(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        Get IPOs with optional filters
        
        Args:
            status: Filter by Upcoming, Open, or Closed
            category: Filter by Mainboard or SME
            
        Returns:
            List of IPOs
        """
        ipos = self._load_json(self.ipos_file)
        
        # Update statuses
        for ipo in ipos:
            ipo['status'] = self._determine_status(ipo['open_date'], ipo['close_date'])
        
        self._save_json(self.ipos_file, ipos)
        
        # Apply filters
        if status:
            ipos = [i for i in ipos if i['status'] == status]
        
        if category:
            ipos = [i for i in ipos if i['category'] == category]
        
        return ipos
    
    def get_upcoming_ipos(self, days: int = 30) -> List[Dict]:
        """Get IPOs opening in next N days"""
        ipos = self._load_json(self.ipos_file)
        today = datetime.now().date()
        cutoff = today + timedelta(days=days)
        
        upcoming = []
        for ipo in ipos:
            open_date = datetime.strptime(ipo['open_date'], "%Y-%m-%d").date()
            if today <= open_date <= cutoff:
                days_until = (open_date - today).days
                ipo['days_until_open'] = days_until
                upcoming.append(ipo)
        
        # Sort by open date
        upcoming.sort(key=lambda x: x['open_date'])
        
        return upcoming
    
    def analyze_ipo(self, ipo_id: int) -> Dict:
        """
        Analyze an IPO and provide recommendation
        
        Args:
            ipo_id: IPO ID to analyze
            
        Returns:
            Analysis with recommendation
        """
        ipos = self._load_json(self.ipos_file)
        
        ipo = next((i for i in ipos if i['id'] == ipo_id), None)
        if not ipo:
            return {"success": False, "message": "IPO not found"}
        
        # Simple analysis based on available data
        # In real scenario, this would include financials, valuations, etc.
        
        analysis = {
            "company": ipo['company_name'],
            "recommendation": "NEUTRAL",  # Default
            "score": 0,
            "factors": [],
            "risks": [],
            "opportunities": []
        }
        
        # Category analysis
        if ipo['category'] == "Mainboard":
            analysis['score'] += 1
            analysis['factors'].append("Listed on mainboard - better liquidity")
        else:
            analysis['risks'].append("SME category - lower liquidity")
        
        # Issue size analysis
        if "Cr" in ipo['issue_size']:
            size_str = ipo['issue_size'].replace('₹', '').replace('Cr', '').strip()
            try:
                size = float(size_str)
                if size >= 500:
                    analysis['score'] += 1
                    analysis['factors'].append("Large issue size - good for institutional interest")
                elif size < 100:
                    analysis['risks'].append("Small issue size - may have liquidity issues")
            except:
                pass
        
        # Price band analysis
        try:
            price_range = ipo['price_band'].replace('₹', '').split('-')
            if len(price_range) == 2:
                low = float(price_range[0])
                high = float(price_range[1])
                
                if high <= 500:
                    analysis['opportunities'].append("Affordable price point for retail investors")
                    analysis['score'] += 0.5
        except:
            pass
        
        # Determine recommendation
        if analysis['score'] >= 2:
            analysis['recommendation'] = "SUBSCRIBE"
        elif analysis['score'] <= -1:
            analysis['recommendation'] = "AVOID"
        else:
            analysis['recommendation'] = "NEUTRAL"
        
        # Add to IPO record
        ipo['analysis'] = analysis
        self._save_json(self.ipos_file, ipos)
        
        return {
            "success": True,
            "ipo": ipo,
            "analysis": analysis
        }
    
    def set_ipo_reminder(
        self,
        ipo_id: int,
        reminder_date: str,
        reminder_type: str = "OPENING",
        email: Optional[str] = None
    ) -> Dict:
        """
        Set reminder for IPO
        
        Args:
            ipo_id: IPO ID
            reminder_date: Date for reminder (YYYY-MM-DD)
            reminder_type: OPENING, CLOSING, or LISTING
            email: Email for notification
            
        Returns:
            Reminder details
        """
        ipos = self._load_json(self.ipos_file)
        reminders = self._load_json(self.ipo_reminders_file)
        
        ipo = next((i for i in ipos if i['id'] == ipo_id), None)
        if not ipo:
            return {"success": False, "message": "IPO not found"}
        
        reminder = {
            "id": len(reminders) + 1,
            "ipo_id": ipo_id,
            "company_name": ipo['company_name'],
            "reminder_date": reminder_date,
            "reminder_type": reminder_type,
            "email": email,
            "status": "ACTIVE",
            "created_at": datetime.now().isoformat()
        }
        
        reminders.append(reminder)
        self._save_json(self.ipo_reminders_file, reminders)
        
        return {
            "success": True,
            "message": f"Reminder set for {ipo['company_name']}",
            "reminder": reminder
        }
    
    def get_ipo_reminders(self, status: Optional[str] = None) -> List[Dict]:
        """Get IPO reminders"""
        reminders = self._load_json(self.ipo_reminders_file)
        
        if status:
            reminders = [r for r in reminders if r['status'] == status.upper()]
        
        return reminders
    
    def get_ipo_calendar(self, month: Optional[int] = None) -> Dict:
        """
        Get IPO calendar for a month
        
        Args:
            month: Month number (1-12), defaults to current month
            
        Returns:
            Calendar with IPOs organized by date
        """
        if month is None:
            month = datetime.now().month
        
        year = datetime.now().year
        ipos = self._load_json(self.ipos_file)
        
        calendar = {}
        
        for ipo in ipos:
            try:
                open_date = datetime.strptime(ipo['open_date'], "%Y-%m-%d")
                if open_date.month == month and open_date.year == year:
                    date_str = open_date.strftime("%Y-%m-%d")
                    if date_str not in calendar:
                        calendar[date_str] = []
                    calendar[date_str].append({
                        "company": ipo['company_name'],
                        "event": "OPENING",
                        "price_band": ipo['price_band'],
                        "category": ipo['category']
                    })
                
                close_date = datetime.strptime(ipo['close_date'], "%Y-%m-%d")
                if close_date.month == month and close_date.year == year:
                    date_str = close_date.strftime("%Y-%m-%d")
                    if date_str not in calendar:
                        calendar[date_str] = []
                    calendar[date_str].append({
                        "company": ipo['company_name'],
                        "event": "CLOSING",
                        "price_band": ipo['price_band'],
                        "category": ipo['category']
                    })
            except:
                continue
        
        return {
            "month": month,
            "year": year,
            "calendar": calendar,
            "total_events": sum(len(events) for events in calendar.values())
        }