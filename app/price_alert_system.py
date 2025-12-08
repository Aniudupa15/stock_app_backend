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


# price_alert_system.py
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import requests

_NSE_BASE = "https://www.nseindia.com"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com"
}

def _nse_session():
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(_NSE_BASE, timeout=10)
    except Exception:
        pass
    return s

def _latest_close_price(symbol: str) -> Optional[float]:
    symbol_clean = symbol.replace('.NS', '').upper()
    url = f"{_NSE_BASE}/api/quote-equity?symbol={symbol_clean}"
    s = _nse_session()
    try:
        r = s.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        # priceInfo or list fields contain 'lastPrice' or 'lastPrice' under 'priceInfo'
        if not data:
            return None
        price_info = data.get('priceInfo') or data.get('data') or data
        # common key lastPrice or lastTradedPrice
        last = None
        if isinstance(price_info, dict):
            last = price_info.get('lastPrice') or price_info.get('lastTradedPrice') or price_info.get('last')
        elif isinstance(price_info, list) and len(price_info):
            last = price_info[0].get('lastPrice') or price_info[0].get('lastTradedPrice')
        if last is None:
            # try to find nested keys
            for k in ('lastPrice', 'ltP', 'last'):
                if k in data:
                    last = data[k]
                    break
        if last is None:
            # try top-level fields
            last = data.get('priceInfo', {}).get('lastPrice')
        if last is None:
            return None
        return float(last)
    except Exception as e:
        print(f"[price_alert] fetch error {symbol_clean}: {e}")
        return None

class PriceAlertSystem:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.alerts_file = self.data_dir / "price_alerts.json"
        self.triggered_file = self.data_dir / "triggered_alerts.json"
        self._init_files()

    def _init_files(self):
        if not self.alerts_file.exists():
            self._save_json(self.alerts_file, [])
        if not self.triggered_file.exists():
            self._save_json(self.triggered_file, [])

    def _load_json(self, filepath: Path) -> any:
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return []

    def _save_json(self, filepath: Path, data: any):
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[price_alert] save error: {e}")

    def normalize_ticker(self, ticker: str) -> str:
        t = ticker.strip().upper()
        if not t.endswith('.NS') and not t.endswith('.BO'):
            t = f"{t}.NS"
        return t

    def create_alert(self, ticker: str, target_price: float, condition: str, email: Optional[str] = None, notes: Optional[str] = None) -> Dict:
        alerts = self._load_json(self.alerts_file)
        ticker_ns = self.normalize_ticker(ticker)
        current_price = _latest_close_price(ticker_ns) or 0.0
        alert = {
            "id": len(alerts) + 1,
            "ticker": ticker_ns,
            "target_price": target_price,
            "condition": condition.lower(),
            "email": email,
            "notes": notes,
            "current_price_at_creation": current_price,
            "created_at": datetime.now().isoformat(),
            "status": "ACTIVE",
            "triggered": False
        }
        alerts.append(alert)
        self._save_json(self.alerts_file, alerts)
        return {"success": True, "message": f"Alert created for {ticker_ns}", "alert": alert}

    def get_alerts(self, status: Optional[str] = None, ticker: Optional[str] = None) -> List[Dict]:
        alerts = self._load_json(self.alerts_file)
        if status:
            alerts = [a for a in alerts if a['status'] == status.upper()]
        if ticker:
            ticker_ns = self.normalize_ticker(ticker)
            alerts = [a for a in alerts if a['ticker'] == ticker_ns]
        return alerts

    def delete_alert(self, alert_id: int) -> Dict:
        alerts = self._load_json(self.alerts_file)
        original_len = len(alerts)
        alerts = [a for a in alerts if a['id'] != alert_id]
        if len(alerts) == original_len:
            return {"success": False, "message": f"Alert {alert_id} not found"}
        self._save_json(self.alerts_file, alerts)
        return {"success": True, "message": f"Alert {alert_id} deleted"}

    def check_alerts(self) -> List[Dict]:
        alerts = self._load_json(self.alerts_file)
        triggered_alerts = []
        for alert in alerts:
            if alert['status'] != 'ACTIVE' or alert.get('triggered'):
                continue
            try:
                current_price = _latest_close_price(alert['ticker'])
                if current_price is None:
                    continue
                target = alert['target_price']
                condition = alert['condition']
                should_trigger = False
                if condition == 'above' and current_price >= target:
                    should_trigger = True
                elif condition == 'below' and current_price <= target:
                    should_trigger = True
                if should_trigger:
                    alert['triggered'] = True
                    alert['status'] = 'TRIGGERED'
                    alert['triggered_at'] = datetime.now().isoformat()
                    alert['triggered_price'] = current_price
                    triggered_alerts.append(alert)
                    triggered_history = self._load_json(self.triggered_file)
                    triggered_history.append(alert)
                    self._save_json(self.triggered_file, triggered_history)
            except Exception as e:
                print(f"[price_alert] checking error id={alert.get('id')}: {e}")
                continue
        self._save_json(self.alerts_file, alerts)
        return triggered_alerts

    def get_triggered_alerts(self, limit: int = 50) -> List[Dict]:
        hist = self._load_json(self.triggered_file)
        return hist[-limit:]

    def get_alert_summary(self) -> Dict:
        alerts = self._load_json(self.alerts_file)
        triggered = self._load_json(self.triggered_file)
        active_count = sum(1 for a in alerts if a['status'] == 'ACTIVE')
        return {
            "total_alerts": len(alerts),
            "active_alerts": active_count,
            "triggered_alerts": len(triggered),
            "alerts_by_stock": self._group_by_ticker(alerts)
        }

    def _group_by_ticker(self, alerts: List[Dict]) -> Dict:
        grouped = {}
        for a in alerts:
            grouped.setdefault(a['ticker'], []).append(a)
        return grouped
