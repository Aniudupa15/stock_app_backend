# import json
# from pathlib import Path
# from typing import List, Dict, Optional
# from datetime import datetime
# import yfinance as yf

# class PriceAlertSystem:
#     """
#     Price alert monitoring system for stock price targets
#     """
    
#     def __init__(self, data_dir: str = "data"):
#         self.data_dir = Path(data_dir)
#         self.data_dir.mkdir(exist_ok=True)
#         self.alerts_file = self.data_dir / "price_alerts.json"
#         self.triggered_file = self.data_dir / "triggered_alerts.json"
        
#         self._init_files()
    
#     def _init_files(self):
#         """Initialize JSON files"""
#         if not self.alerts_file.exists():
#             self._save_json(self.alerts_file, [])
        
#         if not self.triggered_file.exists():
#             self._save_json(self.triggered_file, [])
    
#     def _load_json(self, filepath: Path) -> any:
#         """Load JSON file"""
#         try:
#             with open(filepath, 'r') as f:
#                 return json.load(f)
#         except:
#             return []
    
#     def _save_json(self, filepath: Path, data: any):
#         """Save to JSON file"""
#         try:
#             with open(filepath, 'w') as f:
#                 json.dump(data, f, indent=2)
#         except Exception as e:
#             print(f"Error saving {filepath}: {e}")
    
#     def normalize_ticker(self, ticker: str) -> str:
#         """Add .NS suffix if not present"""
#         ticker = ticker.strip().upper()
#         if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
#             ticker = f"{ticker}.NS"
#         return ticker
    
#     def create_alert(
#         self, 
#         ticker: str, 
#         target_price: float,
#         condition: str,  # "above" or "below"
#         email: Optional[str] = None,
#         notes: Optional[str] = None
#     ) -> Dict:
#         """
#         Create a price alert
        
#         Args:
#             ticker: Stock symbol
#             target_price: Price to trigger alert
#             condition: "above" or "below"
#             email: Email to send alert (optional)
#             notes: Custom notes
            
#         Returns:
#             Created alert details
#         """
#         alerts = self._load_json(self.alerts_file)
        
#         ticker = self.normalize_ticker(ticker)
        
#         # Get current price
#         try:
#             stock = yf.Ticker(ticker)
#             data = stock.history(period="1d")
#             current_price = float(data['Close'].iloc[-1]) if not data.empty else 0
#         except:
#             current_price = 0
        
#         alert = {
#             "id": len(alerts) + 1,
#             "ticker": ticker,
#             "target_price": target_price,
#             "condition": condition.lower(),
#             "email": email,
#             "notes": notes,
#             "current_price_at_creation": current_price,
#             "created_at": datetime.now().isoformat(),
#             "status": "ACTIVE",
#             "triggered": False
#         }
        
#         alerts.append(alert)
#         self._save_json(self.alerts_file, alerts)
        
#         return {
#             "success": True,
#             "message": f"Alert created for {ticker}",
#             "alert": alert
#         }
    
#     def get_alerts(self, status: Optional[str] = None, ticker: Optional[str] = None) -> List[Dict]:
#         """
#         Get all alerts with optional filters
        
#         Args:
#             status: Filter by ACTIVE or TRIGGERED
#             ticker: Filter by stock symbol
            
#         Returns:
#             List of alerts
#         """
#         alerts = self._load_json(self.alerts_file)
        
#         # Apply filters
#         if status:
#             alerts = [a for a in alerts if a['status'] == status.upper()]
        
#         if ticker:
#             ticker = self.normalize_ticker(ticker)
#             alerts = [a for a in alerts if a['ticker'] == ticker]
        
#         return alerts
    
#     def delete_alert(self, alert_id: int) -> Dict:
#         """Delete an alert"""
#         alerts = self._load_json(self.alerts_file)
#         original_length = len(alerts)
        
#         alerts = [a for a in alerts if a['id'] != alert_id]
        
#         if len(alerts) == original_length:
#             return {
#                 "success": False,
#                 "message": f"Alert {alert_id} not found"
#             }
        
#         self._save_json(self.alerts_file, alerts)
        
#         return {
#             "success": True,
#             "message": f"Alert {alert_id} deleted"
#         }
    
#     def check_alerts(self) -> List[Dict]:
#         """
#         Check all active alerts and trigger if conditions met
        
#         Returns:
#             List of triggered alerts
#         """
#         alerts = self._load_json(self.alerts_file)
#         triggered_alerts = []
        
#         for alert in alerts:
#             if alert['status'] != 'ACTIVE' or alert['triggered']:
#                 continue
            
#             try:
#                 # Get current price
#                 ticker = alert['ticker']
#                 stock = yf.Ticker(ticker)
#                 data = stock.history(period="1d")
                
#                 if data.empty:
#                     continue
                
#                 current_price = float(data['Close'].iloc[-1])
#                 target_price = alert['target_price']
#                 condition = alert['condition']
                
#                 # Check if alert should trigger
#                 should_trigger = False
                
#                 if condition == "above" and current_price >= target_price:
#                     should_trigger = True
#                 elif condition == "below" and current_price <= target_price:
#                     should_trigger = True
                
#                 if should_trigger:
#                     # Update alert status
#                     alert['triggered'] = True
#                     alert['status'] = 'TRIGGERED'
#                     alert['triggered_at'] = datetime.now().isoformat()
#                     alert['triggered_price'] = current_price
                    
#                     triggered_alerts.append(alert)
                    
#                     # Save to triggered history
#                     triggered_history = self._load_json(self.triggered_file)
#                     triggered_history.append(alert)
#                     self._save_json(self.triggered_file, triggered_history)
            
#             except Exception as e:
#                 print(f"Error checking alert {alert['id']}: {e}")
#                 continue
        
#         # Save updated alerts
#         self._save_json(self.alerts_file, alerts)
        
#         return triggered_alerts
    
#     def get_triggered_alerts(self, limit: int = 50) -> List[Dict]:
#         """Get history of triggered alerts"""
#         triggered = self._load_json(self.triggered_file)
#         return triggered[-limit:]
    
#     def get_alert_summary(self) -> Dict:
#         """Get summary of all alerts"""
#         alerts = self._load_json(self.alerts_file)
#         triggered = self._load_json(self.triggered_file)
        
#         active_count = sum(1 for a in alerts if a['status'] == 'ACTIVE')
#         triggered_count = len(triggered)
        
#         return {
#             "total_alerts": len(alerts),
#             "active_alerts": active_count,
#             "triggered_alerts": triggered_count,
#             "alerts_by_stock": self._group_by_ticker(alerts)
#         }
    
#     def _group_by_ticker(self, alerts: List[Dict]) -> Dict:
#         """Group alerts by ticker"""
#         grouped = {}
#         for alert in alerts:
#             ticker = alert['ticker']
#             if ticker not in grouped:
#                 grouped[ticker] = []
#             grouped[ticker].append(alert)
#         return grouped




# app/price_alert_system.py
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import requests
import threading

class PriceAlertSystem:
    """
    Cloud-safe Price alert monitoring system.
    - Uses a requests.Session with User-Agent for Yahoo endpoints.
    - Falls back to Yahoo Chart API if direct methods fail.
    - Simple in-memory cache to reduce cloud requests.
    - File-backed alerts storage (still ephemeral on some hosts).
    """

    CACHE_TTL_SECONDS = 60  # cache latest price for 60 seconds

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.alerts_file = self.data_dir / "price_alerts.json"
        self.triggered_file = self.data_dir / "triggered_alerts.json"

        self._init_files()

        # network session with a realistic User-Agent to reduce blocking
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        })
        # simple thread-safe cache for latest prices
        self._price_cache = {}
        self._cache_lock = threading.Lock()

    # -------------------------
    # File helpers
    # -------------------------
    def _init_files(self):
        if not self.alerts_file.exists():
            self._save_json(self.alerts_file, [])
        if not self.triggered_file.exists():
            self._save_json(self.triggered_file, [])

    def _load_json(self, filepath: Path) -> any:
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_json(self, filepath: Path, data: any):
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            # Do not crash the service because of logging failure
            print(f"Error saving {filepath}: {e}")

    # -------------------------
    # Ticker normalization
    # -------------------------
    def normalize_ticker(self, ticker: str) -> str:
        t = ticker.strip().upper()
        if not t.endswith(".NS") and not t.endswith(".BO"):
            t = f"{t}.NS"
        return t

    # -------------------------
    # Cloud-safe price fetcher
    # -------------------------
    def _cache_set(self, key: str, value: float):
        with self._cache_lock:
            self._price_cache[key] = (datetime.utcnow(), value)

    def _cache_get(self, key: str) -> Optional[float]:
        with self._cache_lock:
            item = self._price_cache.get(key)
            if not item:
                return None
            ts, val = item
            if datetime.utcnow() - ts > timedelta(seconds=self.CACHE_TTL_SECONDS):
                # expired
                del self._price_cache[key]
                return None
            return val

    def _fetch_latest_price(self, ticker: str) -> Optional[float]:
        """
        Fetch the latest close price for `ticker` using:
         1) Yahoo Chart API (query1.finance.yahoo.com)
        Returns None on failure.
        """
        key = f"price:{ticker}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        # Attempt Yahoo Chart API (compact, cloud-friendly)
        try:
            url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                f"?interval=1d&range=2d"
            )
            resp = self._session.get(url, timeout=6)
            resp.raise_for_status()
            data = resp.json()

            result = data.get("chart", {}).get("result")
            if not result:
                return None

            result0 = result[0]
            indic = result0.get("indicators", {}).get("quote", [])
            timestamps = result0.get("timestamp", [])
            if not indic or not timestamps:
                return None

            quote = indic[0]
            closes = quote.get("close", [])
            # find last non-null close from the end
            for v in reversed(closes):
                if v is not None:
                    price = float(v)
                    self._cache_set(key, price)
                    return price
        except Exception:
            # swallow and return None (caller will handle)
            return None

        return None

    # -------------------------
    # CRUD for alerts
    # -------------------------
    def create_alert(
        self,
        ticker: str,
        target_price: float,
        condition: str,
        email: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict:
        alerts = self._load_json(self.alerts_file)
        ticker_ns = self.normalize_ticker(ticker)
        condition = condition.lower() if isinstance(condition, str) else "above"

        # safe fetch of current price (may return None -> saved as 0)
        current_price = self._fetch_latest_price(ticker_ns) or 0.0

        alert = {
            "id": len(alerts) + 1,
            "ticker": ticker_ns,
            "target_price": float(target_price),
            "condition": condition,
            "email": email,
            "notes": notes,
            "current_price_at_creation": round(current_price, 2),
            "created_at": datetime.utcnow().isoformat(),
            "status": "ACTIVE",
            "triggered": False,
        }

        alerts.append(alert)
        self._save_json(self.alerts_file, alerts)

        return {"success": True, "message": f"Alert created for {ticker_ns}", "alert": alert}

    def get_alerts(self, status: Optional[str] = None, ticker: Optional[str] = None) -> List[Dict]:
        alerts = self._load_json(self.alerts_file)
        if status:
            alerts = [a for a in alerts if a.get("status", "").upper() == status.upper()]
        if ticker:
            ticker_ns = self.normalize_ticker(ticker)
            alerts = [a for a in alerts if a.get("ticker") == ticker_ns]
        return alerts

    def delete_alert(self, alert_id: int) -> Dict:
        alerts = self._load_json(self.alerts_file)
        original_length = len(alerts)
        alerts = [a for a in alerts if a.get("id") != alert_id]
        if len(alerts) == original_length:
            return {"success": False, "message": f"Alert {alert_id} not found"}
        # reassign ids to keep them contiguous (optional)
        for idx, a in enumerate(alerts, start=1):
            a["id"] = idx
        self._save_json(self.alerts_file, alerts)
        return {"success": True, "message": f"Alert {alert_id} deleted"}

    # -------------------------
    # Checking alerts (safe)
    # -------------------------
    def check_alerts(self) -> List[Dict]:
        """
        Check active alerts and mark those that should trigger.
        Returns list of alerts that were triggered during this run.
        """
        alerts = self._load_json(self.alerts_file)
        triggered_alerts: List[Dict] = []

        for alert in alerts:
            try:
                if alert.get("status") != "ACTIVE" or alert.get("triggered"):
                    continue

                ticker = alert.get("ticker")
                if not ticker:
                    continue

                # use cloud-safe fetcher
                current_price = self._fetch_latest_price(ticker)
                if current_price is None:
                    # cannot fetch price now — skip this alert (prevents false triggers/404s)
                    continue

                target_price = float(alert.get("target_price", 0))
                condition = str(alert.get("condition", "above")).lower()

                should_trigger = False
                if condition == "above" and current_price >= target_price:
                    should_trigger = True
                elif condition == "below" and current_price <= target_price:
                    should_trigger = True

                if should_trigger:
                    alert["triggered"] = True
                    alert["status"] = "TRIGGERED"
                    alert["triggered_at"] = datetime.utcnow().isoformat()
                    alert["triggered_price"] = round(current_price, 2)

                    triggered_alerts.append(alert)

                    # append to triggered history file
                    triggered_history = self._load_json(self.triggered_file)
                    triggered_history.append(alert)
                    # keep history size bounded (optional)
                    if len(triggered_history) > 2000:
                        triggered_history = triggered_history[-2000:]
                    self._save_json(self.triggered_file, triggered_history)

            except Exception as e:
                # keep loop robust; log the issue
                print(f"Error checking alert id={alert.get('id')}: {e}")
                continue

        # save updated alerts back to disk
        self._save_json(self.alerts_file, alerts)
        return triggered_alerts

    # -------------------------
    # Utilities & summaries
    # -------------------------
    def get_triggered_alerts(self, limit: int = 50) -> List[Dict]:
        triggered = self._load_json(self.triggered_file)
        return triggered[-limit:]

    def get_alert_summary(self) -> Dict:
        alerts = self._load_json(self.alerts_file)
        triggered = self._load_json(self.triggered_file)
        active_count = sum(1 for a in alerts if a.get("status") == "ACTIVE")
        triggered_count = len(triggered)
        return {
            "total_alerts": len(alerts),
            "active_alerts": active_count,
            "triggered_alerts": triggered_count,
            "alerts_by_stock": self._group_by_ticker(alerts),
        }

    def _group_by_ticker(self, alerts: List[Dict]) -> Dict:
        grouped: Dict[str, List[Dict]] = {}
        for alert in alerts:
            t = alert.get("ticker", "UNKNOWN")
            grouped.setdefault(t, []).append(alert)
        return grouped
