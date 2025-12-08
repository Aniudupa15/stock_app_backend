import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

class PortfolioManager:
    """
    Portfolio management system for tracking stocks
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.watchlist_file = self.data_dir / "watchlist.json"
        self.portfolio_file = self.data_dir / "portfolio.json"
        self.transactions_file = self.data_dir / "transactions.json"
        
        # Initialize files if they don't exist
        self._init_files()
    
    def _init_files(self):
        """Initialize JSON files if they don't exist"""
        if not self.watchlist_file.exists():
            self._save_json(self.watchlist_file, [])
        
        if not self.portfolio_file.exists():
            self._save_json(self.portfolio_file, [])
        
        if not self.transactions_file.exists():
            self._save_json(self.transactions_file, [])
    
    def _load_json(self, filepath: Path) -> any:
        """Load JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return [] if filepath != self.portfolio_file else []
    
    def _save_json(self, filepath: Path, data: any):
        """Save to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving {filepath}: {e}")
    
    # ==================== WATCHLIST ====================
    
    def add_to_watchlist(self, ticker: str, target_price: Optional[float] = None, 
                         notes: Optional[str] = None) -> Dict:
        """Add stock to watchlist"""
        watchlist = self._load_json(self.watchlist_file)
        
        # Check if already exists
        for item in watchlist:
            if item['ticker'] == ticker.upper():
                return {
                    "success": False,
                    "message": f"{ticker} is already in watchlist",
                    "watchlist": watchlist
                }
        
        entry = {
            "ticker": ticker.upper(),
            "added_date": datetime.now().isoformat(),
            "target_price": target_price,
            "notes": notes
        }
        
        watchlist.append(entry)
        self._save_json(self.watchlist_file, watchlist)
        
        return {
            "success": True,
            "message": f"{ticker} added to watchlist",
            "watchlist": watchlist
        }
    
    def remove_from_watchlist(self, ticker: str) -> Dict:
        """Remove stock from watchlist"""
        watchlist = self._load_json(self.watchlist_file)
        original_length = len(watchlist)
        
        watchlist = [item for item in watchlist if item['ticker'] != ticker.upper()]
        
        if len(watchlist) == original_length:
            return {
                "success": False,
                "message": f"{ticker} not found in watchlist",
                "watchlist": watchlist
            }
        
        self._save_json(self.watchlist_file, watchlist)
        
        return {
            "success": True,
            "message": f"{ticker} removed from watchlist",
            "watchlist": watchlist
        }
    
    def get_watchlist(self) -> List[Dict]:
        """Get all watchlist items"""
        return self._load_json(self.watchlist_file)
    
    # ==================== PORTFOLIO ====================
    
    def add_to_portfolio(self, ticker: str, quantity: int, buy_price: float,
                        buy_date: Optional[str] = None, notes: Optional[str] = None) -> Dict:
        """Add stock to portfolio"""
        portfolio = self._load_json(self.portfolio_file)
        transactions = self._load_json(self.transactions_file)
        
        if buy_date is None:
            buy_date = datetime.now().isoformat()
        
        # Check if stock already exists
        existing = None
        for item in portfolio:
            if item['ticker'] == ticker.upper():
                existing = item
                break
        
        if existing:
            # Update existing position (average price)
            total_quantity = existing['quantity'] + quantity
            total_value = (existing['quantity'] * existing['avg_buy_price']) + (quantity * buy_price)
            new_avg_price = total_value / total_quantity
            
            existing['quantity'] = total_quantity
            existing['avg_buy_price'] = round(new_avg_price, 2)
            existing['last_updated'] = datetime.now().isoformat()
        else:
            # Add new position
            entry = {
                "ticker": ticker.upper(),
                "quantity": quantity,
                "avg_buy_price": buy_price,
                "buy_date": buy_date,
                "last_updated": datetime.now().isoformat(),
                "notes": notes
            }
            portfolio.append(entry)
        
        # Record transaction
        transaction = {
            "type": "BUY",
            "ticker": ticker.upper(),
            "quantity": quantity,
            "price": buy_price,
            "date": buy_date,
            "total_value": round(quantity * buy_price, 2),
            "notes": notes
        }
        transactions.append(transaction)
        
        self._save_json(self.portfolio_file, portfolio)
        self._save_json(self.transactions_file, transactions)
        
        return {
            "success": True,
            "message": f"Added {quantity} shares of {ticker} to portfolio",
            "portfolio": portfolio
        }
    
    def sell_from_portfolio(self, ticker: str, quantity: int, sell_price: float,
                           sell_date: Optional[str] = None, notes: Optional[str] = None) -> Dict:
        """Sell stock from portfolio"""
        portfolio = self._load_json(self.portfolio_file)
        transactions = self._load_json(self.transactions_file)
        
        if sell_date is None:
            sell_date = datetime.now().isoformat()
        
        # Find the stock
        stock = None
        for item in portfolio:
            if item['ticker'] == ticker.upper():
                stock = item
                break
        
        if not stock:
            return {
                "success": False,
                "message": f"{ticker} not found in portfolio",
                "portfolio": portfolio
            }
        
        if stock['quantity'] < quantity:
            return {
                "success": False,
                "message": f"Insufficient quantity. You have {stock['quantity']} shares.",
                "portfolio": portfolio
            }
        
        # Calculate profit/loss
        buy_value = quantity * stock['avg_buy_price']
        sell_value = quantity * sell_price
        profit_loss = sell_value - buy_value
        profit_loss_pct = (profit_loss / buy_value) * 100
        
        # Update portfolio
        stock['quantity'] -= quantity
        stock['last_updated'] = datetime.now().isoformat()
        
        # Remove if quantity is 0
        if stock['quantity'] == 0:
            portfolio = [item for item in portfolio if item['ticker'] != ticker.upper()]
        
        # Record transaction
        transaction = {
            "type": "SELL",
            "ticker": ticker.upper(),
            "quantity": quantity,
            "price": sell_price,
            "date": sell_date,
            "total_value": round(sell_value, 2),
            "profit_loss": round(profit_loss, 2),
            "profit_loss_pct": round(profit_loss_pct, 2),
            "notes": notes
        }
        transactions.append(transaction)
        
        self._save_json(self.portfolio_file, portfolio)
        self._save_json(self.transactions_file, transactions)
        
        return {
            "success": True,
            "message": f"Sold {quantity} shares of {ticker}",
            "profit_loss": round(profit_loss, 2),
            "profit_loss_pct": round(profit_loss_pct, 2),
            "portfolio": portfolio
        }
    
    def get_portfolio(self) -> List[Dict]:
        """Get all portfolio holdings"""
        return self._load_json(self.portfolio_file)
    
    def get_transactions(self, ticker: Optional[str] = None, 
                        transaction_type: Optional[str] = None) -> List[Dict]:
        """Get transaction history"""
        transactions = self._load_json(self.transactions_file)
        
        # Filter by ticker if provided
        if ticker:
            transactions = [t for t in transactions if t['ticker'] == ticker.upper()]
        
        # Filter by type if provided
        if transaction_type:
            transactions = [t for t in transactions if t['type'] == transaction_type.upper()]
        
        return transactions
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate total portfolio value and P&L"""
        portfolio = self._load_json(self.portfolio_file)
        
        total_investment = 0
        total_current_value = 0
        holdings_detail = []
        
        for holding in portfolio:
            ticker = holding['ticker']
            quantity = holding['quantity']
            avg_buy_price = holding['avg_buy_price']
            
            investment = quantity * avg_buy_price
            current_price = current_prices.get(ticker, avg_buy_price)
            current_value = quantity * current_price
            
            profit_loss = current_value - investment
            profit_loss_pct = (profit_loss / investment) * 100 if investment > 0 else 0
            
            holdings_detail.append({
                "ticker": ticker,
                "quantity": quantity,
                "avg_buy_price": round(avg_buy_price, 2),
                "current_price": round(current_price, 2),
                "investment": round(investment, 2),
                "current_value": round(current_value, 2),
                "profit_loss": round(profit_loss, 2),
                "profit_loss_pct": round(profit_loss_pct, 2)
            })
            
            total_investment += investment
            total_current_value += current_value
        
        total_profit_loss = total_current_value - total_investment
        total_profit_loss_pct = (total_profit_loss / total_investment) * 100 if total_investment > 0 else 0
        
        return {
            "total_investment": round(total_investment, 2),
            "total_current_value": round(total_current_value, 2),
            "total_profit_loss": round(total_profit_loss, 2),
            "total_profit_loss_pct": round(total_profit_loss_pct, 2),
            "holdings": holdings_detail,
            "number_of_holdings": len(holdings_detail)
        }
    
    def clear_portfolio(self) -> Dict:
        """Clear all portfolio data (use with caution!)"""
        self._save_json(self.portfolio_file, [])
        return {
            "success": True,
            "message": "Portfolio cleared"
        }
    
    def clear_watchlist(self) -> Dict:
        """Clear all watchlist data"""
        self._save_json(self.watchlist_file, [])
        return {
            "success": True,
            "message": "Watchlist cleared"
        }