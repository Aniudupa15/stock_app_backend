# import json
# import os
# from typing import List, Dict, Optional
# from datetime import datetime
# from pathlib import Path

# class PortfolioManager:
#     """
#     Portfolio management system for tracking stocks
#     """
    
#     def __init__(self, data_dir: str = "data"):
#         self.data_dir = Path(data_dir)
#         self.data_dir.mkdir(exist_ok=True)
#         self.watchlist_file = self.data_dir / "watchlist.json"
#         self.portfolio_file = self.data_dir / "portfolio.json"
#         self.transactions_file = self.data_dir / "transactions.json"
        
#         # Initialize files if they don't exist
#         self._init_files()
    
#     def _init_files(self):
#         """Initialize JSON files if they don't exist"""
#         if not self.watchlist_file.exists():
#             self._save_json(self.watchlist_file, [])
        
#         if not self.portfolio_file.exists():
#             self._save_json(self.portfolio_file, [])
        
#         if not self.transactions_file.exists():
#             self._save_json(self.transactions_file, [])
    
#     def _load_json(self, filepath: Path) -> any:
#         """Load JSON file"""
#         try:
#             with open(filepath, 'r') as f:
#                 return json.load(f)
#         except Exception as e:
#             print(f"Error loading {filepath}: {e}")
#             return [] if filepath != self.portfolio_file else []
    
#     def _save_json(self, filepath: Path, data: any):
#         """Save to JSON file"""
#         try:
#             with open(filepath, 'w') as f:
#                 json.dump(data, f, indent=2)
#         except Exception as e:
#             print(f"Error saving {filepath}: {e}")
    
#     # ==================== WATCHLIST ====================
    
#     def add_to_watchlist(self, ticker: str, target_price: Optional[float] = None, 
#                          notes: Optional[str] = None) -> Dict:
#         """Add stock to watchlist"""
#         watchlist = self._load_json(self.watchlist_file)
        
#         # Check if already exists
#         for item in watchlist:
#             if item['ticker'] == ticker.upper():
#                 return {
#                     "success": False,
#                     "message": f"{ticker} is already in watchlist",
#                     "watchlist": watchlist
#                 }
        
#         entry = {
#             "ticker": ticker.upper(),
#             "added_date": datetime.now().isoformat(),
#             "target_price": target_price,
#             "notes": notes
#         }
        
#         watchlist.append(entry)
#         self._save_json(self.watchlist_file, watchlist)
        
#         return {
#             "success": True,
#             "message": f"{ticker} added to watchlist",
#             "watchlist": watchlist
#         }
    
#     def remove_from_watchlist(self, ticker: str) -> Dict:
#         """Remove stock from watchlist"""
#         watchlist = self._load_json(self.watchlist_file)
#         original_length = len(watchlist)
        
#         watchlist = [item for item in watchlist if item['ticker'] != ticker.upper()]
        
#         if len(watchlist) == original_length:
#             return {
#                 "success": False,
#                 "message": f"{ticker} not found in watchlist",
#                 "watchlist": watchlist
#             }
        
#         self._save_json(self.watchlist_file, watchlist)
        
#         return {
#             "success": True,
#             "message": f"{ticker} removed from watchlist",
#             "watchlist": watchlist
#         }
    
#     def get_watchlist(self) -> List[Dict]:
#         """Get all watchlist items"""
#         return self._load_json(self.watchlist_file)
    
#     # ==================== PORTFOLIO ====================
    
#     def add_to_portfolio(self, ticker: str, quantity: int, buy_price: float,
#                         buy_date: Optional[str] = None, notes: Optional[str] = None) -> Dict:
#         """Add stock to portfolio"""
#         portfolio = self._load_json(self.portfolio_file)
#         transactions = self._load_json(self.transactions_file)
        
#         if buy_date is None:
#             buy_date = datetime.now().isoformat()
        
#         # Check if stock already exists
#         existing = None
#         for item in portfolio:
#             if item['ticker'] == ticker.upper():
#                 existing = item
#                 break
        
#         if existing:
#             # Update existing position (average price)
#             total_quantity = existing['quantity'] + quantity
#             total_value = (existing['quantity'] * existing['avg_buy_price']) + (quantity * buy_price)
#             new_avg_price = total_value / total_quantity
            
#             existing['quantity'] = total_quantity
#             existing['avg_buy_price'] = round(new_avg_price, 2)
#             existing['last_updated'] = datetime.now().isoformat()
#         else:
#             # Add new position
#             entry = {
#                 "ticker": ticker.upper(),
#                 "quantity": quantity,
#                 "avg_buy_price": buy_price,
#                 "buy_date": buy_date,
#                 "last_updated": datetime.now().isoformat(),
#                 "notes": notes
#             }
#             portfolio.append(entry)
        
#         # Record transaction
#         transaction = {
#             "type": "BUY",
#             "ticker": ticker.upper(),
#             "quantity": quantity,
#             "price": buy_price,
#             "date": buy_date,
#             "total_value": round(quantity * buy_price, 2),
#             "notes": notes
#         }
#         transactions.append(transaction)
        
#         self._save_json(self.portfolio_file, portfolio)
#         self._save_json(self.transactions_file, transactions)
        
#         return {
#             "success": True,
#             "message": f"Added {quantity} shares of {ticker} to portfolio",
#             "portfolio": portfolio
#         }
    
#     def sell_from_portfolio(self, ticker: str, quantity: int, sell_price: float,
#                            sell_date: Optional[str] = None, notes: Optional[str] = None) -> Dict:
#         """Sell stock from portfolio"""
#         portfolio = self._load_json(self.portfolio_file)
#         transactions = self._load_json(self.transactions_file)
        
#         if sell_date is None:
#             sell_date = datetime.now().isoformat()
        
#         # Find the stock
#         stock = None
#         for item in portfolio:
#             if item['ticker'] == ticker.upper():
#                 stock = item
#                 break
        
#         if not stock:
#             return {
#                 "success": False,
#                 "message": f"{ticker} not found in portfolio",
#                 "portfolio": portfolio
#             }
        
#         if stock['quantity'] < quantity:
#             return {
#                 "success": False,
#                 "message": f"Insufficient quantity. You have {stock['quantity']} shares.",
#                 "portfolio": portfolio
#             }
        
#         # Calculate profit/loss
#         buy_value = quantity * stock['avg_buy_price']
#         sell_value = quantity * sell_price
#         profit_loss = sell_value - buy_value
#         profit_loss_pct = (profit_loss / buy_value) * 100
        
#         # Update portfolio
#         stock['quantity'] -= quantity
#         stock['last_updated'] = datetime.now().isoformat()
        
#         # Remove if quantity is 0
#         if stock['quantity'] == 0:
#             portfolio = [item for item in portfolio if item['ticker'] != ticker.upper()]
        
#         # Record transaction
#         transaction = {
#             "type": "SELL",
#             "ticker": ticker.upper(),
#             "quantity": quantity,
#             "price": sell_price,
#             "date": sell_date,
#             "total_value": round(sell_value, 2),
#             "profit_loss": round(profit_loss, 2),
#             "profit_loss_pct": round(profit_loss_pct, 2),
#             "notes": notes
#         }
#         transactions.append(transaction)
        
#         self._save_json(self.portfolio_file, portfolio)
#         self._save_json(self.transactions_file, transactions)
        
#         return {
#             "success": True,
#             "message": f"Sold {quantity} shares of {ticker}",
#             "profit_loss": round(profit_loss, 2),
#             "profit_loss_pct": round(profit_loss_pct, 2),
#             "portfolio": portfolio
#         }
    
#     def get_portfolio(self) -> List[Dict]:
#         """Get all portfolio holdings"""
#         return self._load_json(self.portfolio_file)
    
#     def get_transactions(self, ticker: Optional[str] = None, 
#                         transaction_type: Optional[str] = None) -> List[Dict]:
#         """Get transaction history"""
#         transactions = self._load_json(self.transactions_file)
        
#         # Filter by ticker if provided
#         if ticker:
#             transactions = [t for t in transactions if t['ticker'] == ticker.upper()]
        
#         # Filter by type if provided
#         if transaction_type:
#             transactions = [t for t in transactions if t['type'] == transaction_type.upper()]
        
#         return transactions
    
#     def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> Dict:
#         """Calculate total portfolio value and P&L"""
#         portfolio = self._load_json(self.portfolio_file)
        
#         total_investment = 0
#         total_current_value = 0
#         holdings_detail = []
        
#         for holding in portfolio:
#             ticker = holding['ticker']
#             quantity = holding['quantity']
#             avg_buy_price = holding['avg_buy_price']
            
#             investment = quantity * avg_buy_price
#             current_price = current_prices.get(ticker, avg_buy_price)
#             current_value = quantity * current_price
            
#             profit_loss = current_value - investment
#             profit_loss_pct = (profit_loss / investment) * 100 if investment > 0 else 0
            
#             holdings_detail.append({
#                 "ticker": ticker,
#                 "quantity": quantity,
#                 "avg_buy_price": round(avg_buy_price, 2),
#                 "current_price": round(current_price, 2),
#                 "investment": round(investment, 2),
#                 "current_value": round(current_value, 2),
#                 "profit_loss": round(profit_loss, 2),
#                 "profit_loss_pct": round(profit_loss_pct, 2)
#             })
            
#             total_investment += investment
#             total_current_value += current_value
        
#         total_profit_loss = total_current_value - total_investment
#         total_profit_loss_pct = (total_profit_loss / total_investment) * 100 if total_investment > 0 else 0
        
#         return {
#             "total_investment": round(total_investment, 2),
#             "total_current_value": round(total_current_value, 2),
#             "total_profit_loss": round(total_profit_loss, 2),
#             "total_profit_loss_pct": round(total_profit_loss_pct, 2),
#             "holdings": holdings_detail,
#             "number_of_holdings": len(holdings_detail)
#         }
    
#     def clear_portfolio(self) -> Dict:
#         """Clear all portfolio data (use with caution!)"""
#         self._save_json(self.portfolio_file, [])
#         return {
#             "success": True,
#             "message": "Portfolio cleared"
#         }
    
#     def clear_watchlist(self) -> Dict:
#         """Clear all watchlist data"""
#         self._save_json(self.watchlist_file, [])
#         return {
#             "success": True,
#             "message": "Watchlist cleared"
#         }



import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path


class PortfolioManager:
    """
    Cloud-safe Portfolio Management System
    Compatible with Render, HuggingFace, Railway, Docker, etc.
    """

    def __init__(self, data_dir: str = None):
        # Use /tmp for cloud-safe writes
        if data_dir is None:
            # Render, HuggingFace & serverless environments always allow /tmp writes
            data_dir = "/tmp/portfolio_data"

        self.data_dir = Path(data_dir)

        # Best-effort directory creation
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.read_only_mode = False
        except Exception:
            # Fallback: read-only environment → operate fully from RAM
            self.read_only_mode = True
            self.memory_store = {
                "watchlist": [],
                "portfolio": [],
                "transactions": []
            }

        # Define file paths
        self.watchlist_file = self.data_dir / "watchlist.json"
        self.portfolio_file = self.data_dir / "portfolio.json"
        self.transactions_file = self.data_dir / "transactions.json"

        self._init_files()

    # -------------------------------------------------------------------------
    # CORE SAFE JSON OPS
    # -------------------------------------------------------------------------

    def _init_files(self):
        """Initialize JSON files or memory storage."""
        if self.read_only_mode:
            return

        for file in [self.watchlist_file, self.portfolio_file, self.transactions_file]:
            if not file.exists():
                self._save_json(file, [])

    def _load_json(self, filepath: Path):
        """Load JSON safely."""
        if self.read_only_mode:
            key = filepath.name.replace(".json", "")
            return self.memory_store.get(key, [])

        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_json(self, filepath: Path, data):
        """Write JSON safely (fallback to RAM if FS fails)."""
        if self.read_only_mode:
            key = filepath.name.replace(".json", "")
            self.memory_store[key] = data
            return

        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            # Fallback to RAM mode if write fails
            self.read_only_mode = True
            key = filepath.name.replace(".json", "")
            self.memory_store[key] = data

    # -------------------------------------------------------------------------
    # WATCHLIST
    # -------------------------------------------------------------------------

    def add_to_watchlist(self, ticker: str, target_price: Optional[float] = None,
                         notes: Optional[str] = None) -> Dict:

        watchlist = self._load_json(self.watchlist_file)
        ticker = ticker.upper()

        # Prevent duplicates
        if any(item["ticker"] == ticker for item in watchlist):
            return {"success": False, "message": f"{ticker} already exists", "watchlist": watchlist}

        entry = {
            "ticker": ticker,
            "added_date": datetime.now().isoformat(),
            "target_price": target_price,
            "notes": notes
        }

        watchlist.append(entry)
        self._save_json(self.watchlist_file, watchlist)

        return {"success": True, "message": f"{ticker} added", "watchlist": watchlist}

    def remove_from_watchlist(self, ticker: str) -> Dict:
        ticker = ticker.upper()
        watchlist = self._load_json(self.watchlist_file)
        updated = [item for item in watchlist if item["ticker"] != ticker]

        if len(updated) == len(watchlist):
            return {"success": False, "message": f"{ticker} not found", "watchlist": watchlist}

        self._save_json(self.watchlist_file, updated)
        return {"success": True, "message": f"{ticker} removed", "watchlist": updated}

    def get_watchlist(self) -> List[Dict]:
        return self._load_json(self.watchlist_file)

    # -------------------------------------------------------------------------
    # PORTFOLIO
    # -------------------------------------------------------------------------

    def add_to_portfolio(self, ticker: str, quantity: int, buy_price: float,
                         buy_date: Optional[str] = None, notes: Optional[str] = None) -> Dict:

        ticker = ticker.upper()
        portfolio = self._load_json(self.portfolio_file)
        transactions = self._load_json(self.transactions_file)

        if not buy_date:
            buy_date = datetime.now().isoformat()

        # Check if position exists
        existing = next((p for p in portfolio if p["ticker"] == ticker), None)

        if existing:
            # Weighted average update
            old_qty = existing["quantity"]
            new_qty = old_qty + quantity
            total_cost = (old_qty * existing["avg_buy_price"]) + (quantity * buy_price)
            existing["quantity"] = new_qty
            existing["avg_buy_price"] = round(total_cost / new_qty, 2)
            existing["last_updated"] = datetime.now().isoformat()
        else:
            portfolio.append({
                "ticker": ticker,
                "quantity": quantity,
                "avg_buy_price": buy_price,
                "buy_date": buy_date,
                "last_updated": datetime.now().isoformat(),
                "notes": notes
            })

        # Log buy transaction
        transactions.append({
            "type": "BUY",
            "ticker": ticker,
            "quantity": quantity,
            "price": buy_price,
            "date": buy_date,
            "total_value": round(quantity * buy_price, 2),
            "notes": notes
        })

        self._save_json(self.portfolio_file, portfolio)
        self._save_json(self.transactions_file, transactions)

        return {"success": True, "message": f"Bought {quantity} x {ticker}", "portfolio": portfolio}

    def sell_from_portfolio(self, ticker: str, quantity: int, sell_price: float,
                            sell_date: Optional[str] = None, notes: Optional[str] = None) -> Dict:

        ticker = ticker.upper()
        portfolio = self._load_json(self.portfolio_file)
        transactions = self._load_json(self.transactions_file)

        if not sell_date:
            sell_date = datetime.now().isoformat()

        stock = next((p for p in portfolio if p["ticker"] == ticker), None)

        if not stock:
            return {"success": False, "message": f"{ticker} not found", "portfolio": portfolio}

        if stock["quantity"] < quantity:
            return {"success": False, "message": f"Not enough quantity", "portfolio": portfolio}

        # P/L calculation
        investment = quantity * stock["avg_buy_price"]
        proceeds = quantity * sell_price
        profit_loss = proceeds - investment
        profit_loss_pct = (profit_loss / investment) * 100

        # Update quantity
        stock["quantity"] -= quantity
        stock["last_updated"] = datetime.now().isoformat()

        if stock["quantity"] == 0:
            portfolio = [p for p in portfolio if p["ticker"] != ticker]

        # Log sell
        transactions.append({
            "type": "SELL",
            "ticker": ticker,
            "quantity": quantity,
            "price": sell_price,
            "date": sell_date,
            "total_value": proceeds,
            "profit_loss": round(profit_loss, 2),
            "profit_loss_pct": round(profit_loss_pct, 2),
            "notes": notes
        })

        self._save_json(self.portfolio_file, portfolio)
        self._save_json(self.transactions_file, transactions)

        return {
            "success": True,
            "message": f"Sold {quantity} x {ticker}",
            "profit_loss": round(profit_loss, 2),
            "profit_loss_pct": round(profit_loss_pct, 2),
            "portfolio": portfolio
        }

    # -------------------------------------------------------------------------
    # READ OPERATIONS
    # -------------------------------------------------------------------------

    def get_portfolio(self) -> List[Dict]:
        return self._load_json(self.portfolio_file)

    def get_transactions(self, ticker: Optional[str] = None,
                         transaction_type: Optional[str] = None) -> List[Dict]:

        transactions = self._load_json(self.transactions_file)

        if ticker:
            transactions = [t for t in transactions if t["ticker"] == ticker.upper()]
        if transaction_type:
            transactions = [t for t in transactions if t["type"] == transaction_type.upper()]

        return transactions

    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> Dict:
        portfolio = self._load_json(self.portfolio_file)

        total_investment, total_value = 0, 0
        details = []

        for p in portfolio:
            qty = p["quantity"]
            buy = p["avg_buy_price"]
            current = current_prices.get(p["ticker"], buy)

            invest = qty * buy
            value = qty * current
            pl = value - invest

            details.append({
                "ticker": p["ticker"],
                "quantity": qty,
                "avg_buy_price": round(buy, 2),
                "current_price": round(current, 2),
                "investment": round(invest, 2),
                "current_value": round(value, 2),
                "profit_loss": round(pl, 2),
                "profit_loss_pct": round((pl / invest) * 100 if invest else 0, 2)
            })

            total_investment += invest
            total_value += value

        total_pl = total_value - total_investment
        total_pl_pct = (total_pl / total_investment) * 100 if total_investment else 0

        return {
            "total_investment": round(total_investment, 2),
            "total_current_value": round(total_value, 2),
            "total_profit_loss": round(total_pl, 2),
            "total_profit_loss_pct": round(total_pl_pct, 2),
            "holdings": details,
            "number_of_holdings": len(details)
        }

    # -------------------------------------------------------------------------
    # ADMIN
    # -------------------------------------------------------------------------

    def clear_portfolio(self):
        self._save_json(self.portfolio_file, [])
        return {"success": True, "message": "Portfolio cleared"}

    def clear_watchlist(self):
        self._save_json(self.watchlist_file, [])
        return {"success": True, "message": "Watchlist cleared"}
