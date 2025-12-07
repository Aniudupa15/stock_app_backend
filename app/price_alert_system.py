import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import yfinance as yf

class PriceAlertSystem:
    """
    Price alert monitoring system for stock price targets
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.alerts_file = self.data_dir / "price_alerts.json"
        self.triggered_file = self.data_dir / "triggered_alerts.json"
        
        self._init_files()
    
    def _init_files(self):
        """Initialize JSON files"""
        if not self.alerts_file.exists():
            self._save_json(self.alerts_file, [])
        
        if not self.triggered_file.exists():
            self._save_json(self.triggered_file, [])
    
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
    
    def normalize_ticker(self, ticker: str) -> str:
        """Add .NS suffix if not present"""
        ticker = ticker.strip().upper()
        if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
            ticker = f"{ticker}.NS"
        return ticker
    
    def create_alert(
        self, 
        ticker: str, 
        target_price: float,
        condition: str,  # "above" or "below"
        email: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Create a price alert
        
        Args:
            ticker: Stock symbol
            target_price: Price to trigger alert
            condition: "above" or "below"
            email: Email to send alert (optional)
            notes: Custom notes
            
        Returns:
            Created alert details
        """
        alerts = self._load_json(self.alerts_file)
        
        ticker = self.normalize_ticker(ticker)
        
        # Get current price
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            current_price = float(data['Close'].iloc[-1]) if not data.empty else 0
        except:
            current_price = 0
        
        alert = {
            "id": len(alerts) + 1,
            "ticker": ticker,
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
        
        return {
            "success": True,
            "message": f"Alert created for {ticker}",
            "alert": alert
        }
    
    def get_alerts(self, status: Optional[str] = None, ticker: Optional[str] = None) -> List[Dict]:
        """
        Get all alerts with optional filters
        
        Args:
            status: Filter by ACTIVE or TRIGGERED
            ticker: Filter by stock symbol
            
        Returns:
            List of alerts
        """
        alerts = self._load_json(self.alerts_file)
        
        # Apply filters
        if status:
            alerts = [a for a in alerts if a['status'] == status.upper()]
        
        if ticker:
            ticker = self.normalize_ticker(ticker)
            alerts = [a for a in alerts if a['ticker'] == ticker]
        
        return alerts
    
    def delete_alert(self, alert_id: int) -> Dict:
        """Delete an alert"""
        alerts = self._load_json(self.alerts_file)
        original_length = len(alerts)
        
        alerts = [a for a in alerts if a['id'] != alert_id]
        
        if len(alerts) == original_length:
            return {
                "success": False,
                "message": f"Alert {alert_id} not found"
            }
        
        self._save_json(self.alerts_file, alerts)
        
        return {
            "success": True,
            "message": f"Alert {alert_id} deleted"
        }
    
    def check_alerts(self) -> List[Dict]:
        """
        Check all active alerts and trigger if conditions met
        
        Returns:
            List of triggered alerts
        """
        alerts = self._load_json(self.alerts_file)
        triggered_alerts = []
        
        for alert in alerts:
            if alert['status'] != 'ACTIVE' or alert['triggered']:
                continue
            
            try:
                # Get current price
                ticker = alert['ticker']
                stock = yf.Ticker(ticker)
                data = stock.history(period="1d")
                
                if data.empty:
                    continue
                
                current_price = float(data['Close'].iloc[-1])
                target_price = alert['target_price']
                condition = alert['condition']
                
                # Check if alert should trigger
                should_trigger = False
                
                if condition == "above" and current_price >= target_price:
                    should_trigger = True
                elif condition == "below" and current_price <= target_price:
                    should_trigger = True
                
                if should_trigger:
                    # Update alert status
                    alert['triggered'] = True
                    alert['status'] = 'TRIGGERED'
                    alert['triggered_at'] = datetime.now().isoformat()
                    alert['triggered_price'] = current_price
                    
                    triggered_alerts.append(alert)
                    
                    # Save to triggered history
                    triggered_history = self._load_json(self.triggered_file)
                    triggered_history.append(alert)
                    self._save_json(self.triggered_file, triggered_history)
            
            except Exception as e:
                print(f"Error checking alert {alert['id']}: {e}")
                continue
        
        # Save updated alerts
        self._save_json(self.alerts_file, alerts)
        
        return triggered_alerts
    
    def get_triggered_alerts(self, limit: int = 50) -> List[Dict]:
        """Get history of triggered alerts"""
        triggered = self._load_json(self.triggered_file)
        return triggered[-limit:]
    
    def get_alert_summary(self) -> Dict:
        """Get summary of all alerts"""
        alerts = self._load_json(self.alerts_file)
        triggered = self._load_json(self.triggered_file)
        
        active_count = sum(1 for a in alerts if a['status'] == 'ACTIVE')
        triggered_count = len(triggered)
        
        return {
            "total_alerts": len(alerts),
            "active_alerts": active_count,
            "triggered_alerts": triggered_count,
            "alerts_by_stock": self._group_by_ticker(alerts)
        }
    
    def _group_by_ticker(self, alerts: List[Dict]) -> Dict:
        """Group alerts by ticker"""
        grouped = {}
        for alert in alerts:
            ticker = alert['ticker']
            if ticker not in grouped:
                grouped[ticker] = []
            grouped[ticker].append(alert)
        return grouped