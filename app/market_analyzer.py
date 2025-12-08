# import yfinance as yf
# import pandas as pd
# import numpy as np
# from typing import List, Dict, Optional
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# class MarketAnalyzer:
#     """
#     Advanced market analysis engine for NSE stocks
#     """
    
#     # Popular NSE stocks for market overview
#     NIFTY_50_STOCKS = [
#         'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
#         'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
#         'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA',
#         'TITAN', 'ULTRACEMCO', 'BAJFINANCE', 'NESTLEIND', 'WIPRO',
#         'ONGC', 'NTPC', 'POWERGRID', 'M&M', 'TECHM',
#         'TATAMOTORS', 'HCLTECH', 'ADANIPORTS', 'COALINDIA', 'TATASTEEL'
#     ]
    
#     def __init__(self):
#         self.cache = {}
#         self.cache_duration = timedelta(minutes=5)
    
#     def normalize_ticker(self, ticker: str) -> str:
#         """Add .NS suffix if not present"""
#         ticker = ticker.strip().upper()
#         if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
#             ticker = f"{ticker}.NS"
#         return ticker
    
#     def get_stock_data(self, ticker: str, period: str = "1d") -> Optional[pd.DataFrame]:
#         """Fetch stock data with caching"""
#         cache_key = f"{ticker}_{period}"
        
#         if cache_key in self.cache:
#             cached_time, cached_data = self.cache[cache_key]
#             if datetime.now() - cached_time < self.cache_duration:
#                 return cached_data
        
#         try:
#             ticker = self.normalize_ticker(ticker)
#             stock = yf.Ticker(ticker)
#             df = stock.history(period=period)
            
#             if not df.empty:
#                 self.cache[cache_key] = (datetime.now(), df)
#                 return df
#         except Exception as e:
#             print(f"Error fetching {ticker}: {e}")
        
#         return None
    
#     def get_top_gainers(self, stocks: List[str] = None, limit: int = 10) -> List[Dict]:
#         """
#         Get top gaining stocks for the day
        
#         Args:
#             stocks: List of stock symbols (default: NIFTY 50)
#             limit: Number of top gainers to return
            
#         Returns:
#             List of dictionaries with stock info
#         """
#         if stocks is None:
#             stocks = self.NIFTY_50_STOCKS
        
#         gainers = []
        
#         for ticker in stocks:
#             df = self.get_stock_data(ticker, period="5d")
#             if df is not None and len(df) >= 2:
#                 try:
#                     current_price = df['Close'].iloc[-1]
#                     prev_close = df['Close'].iloc[-2]
#                     change = current_price - prev_close
#                     change_pct = (change / prev_close) * 100
                    
#                     gainers.append({
#                         'ticker': ticker,
#                         'ticker_ns': self.normalize_ticker(ticker),
#                         'current_price': round(float(current_price), 2),
#                         'previous_close': round(float(prev_close), 2),
#                         'change': round(float(change), 2),
#                         'change_percent': round(float(change_pct), 2),
#                         'volume': int(df['Volume'].iloc[-1]),
#                         'high': round(float(df['High'].iloc[-1]), 2),
#                         'low': round(float(df['Low'].iloc[-1]), 2)
#                     })
#                 except Exception as e:
#                     print(f"Error processing {ticker}: {e}")
#                     continue
        
#         # Sort by change percentage (descending)
#         gainers.sort(key=lambda x: x['change_percent'], reverse=True)
#         return gainers[:limit]
    
#     def get_top_losers(self, stocks: List[str] = None, limit: int = 10) -> List[Dict]:
#         """
#         Get top losing stocks for the day
        
#         Args:
#             stocks: List of stock symbols (default: NIFTY 50)
#             limit: Number of top losers to return
            
#         Returns:
#             List of dictionaries with stock info
#         """
#         if stocks is None:
#             stocks = self.NIFTY_50_STOCKS
        
#         losers = []
        
#         for ticker in stocks:
#             df = self.get_stock_data(ticker, period="5d")
#             if df is not None and len(df) >= 2:
#                 try:
#                     current_price = df['Close'].iloc[-1]
#                     prev_close = df['Close'].iloc[-2]
#                     change = current_price - prev_close
#                     change_pct = (change / prev_close) * 100
                    
#                     losers.append({
#                         'ticker': ticker,
#                         'ticker_ns': self.normalize_ticker(ticker),
#                         'current_price': round(float(current_price), 2),
#                         'previous_close': round(float(prev_close), 2),
#                         'change': round(float(change), 2),
#                         'change_percent': round(float(change_pct), 2),
#                         'volume': int(df['Volume'].iloc[-1]),
#                         'high': round(float(df['High'].iloc[-1]), 2),
#                         'low': round(float(df['Low'].iloc[-1]), 2)
#                     })
#                 except Exception as e:
#                     print(f"Error processing {ticker}: {e}")
#                     continue
        
#         # Sort by change percentage (ascending)
#         losers.sort(key=lambda x: x['change_percent'])
#         return losers[:limit]
    
#     def get_stock_analysis(self, ticker: str, period: str = "1y") -> Optional[Dict]:
#         """
#         Comprehensive stock analysis with technical indicators
        
#         Args:
#             ticker: Stock symbol
#             period: Analysis period
            
#         Returns:
#             Dictionary with comprehensive analysis
#         """
#         df = self.get_stock_data(ticker, period=period)
#         if df is None or df.empty:
#             return None
        
#         try:
#             ticker_ns = self.normalize_ticker(ticker)
#             current_price = float(df['Close'].iloc[-1])
            
#             # Price statistics
#             high_52w = float(df['High'].max())
#             low_52w = float(df['Low'].min())
#             avg_volume = int(df['Volume'].mean())
            
#             # Moving averages
#             df['sma_20'] = df['Close'].rolling(window=20).mean()
#             df['sma_50'] = df['Close'].rolling(window=50).mean()
#             df['sma_200'] = df['Close'].rolling(window=200).mean()
            
#             sma_20 = float(df['sma_20'].iloc[-1]) if not pd.isna(df['sma_20'].iloc[-1]) else None
#             sma_50 = float(df['sma_50'].iloc[-1]) if not pd.isna(df['sma_50'].iloc[-1]) else None
#             sma_200 = float(df['sma_200'].iloc[-1]) if not pd.isna(df['sma_200'].iloc[-1]) else None
            
#             # RSI
#             delta = df['Close'].diff()
#             gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#             loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#             rs = gain / loss.replace(0, np.nan)
#             rsi = 100 - (100 / (1 + rs))
#             current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            
#             # MACD
#             exp1 = df['Close'].ewm(span=12, adjust=False).mean()
#             exp2 = df['Close'].ewm(span=26, adjust=False).mean()
#             macd = exp1 - exp2
#             signal = macd.ewm(span=9, adjust=False).mean()
#             macd_histogram = macd - signal
            
#             current_macd = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0
#             current_signal = float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else 0
#             current_histogram = float(macd_histogram.iloc[-1]) if not pd.isna(macd_histogram.iloc[-1]) else 0
            
#             # Bollinger Bands
#             bb_period = 20
#             bb_std = 2
#             df['bb_middle'] = df['Close'].rolling(window=bb_period).mean()
#             bb_std_dev = df['Close'].rolling(window=bb_period).std()
#             df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
#             df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
            
#             bb_upper = float(df['bb_upper'].iloc[-1]) if not pd.isna(df['bb_upper'].iloc[-1]) else None
#             bb_lower = float(df['bb_lower'].iloc[-1]) if not pd.isna(df['bb_lower'].iloc[-1]) else None
#             bb_middle = float(df['bb_middle'].iloc[-1]) if not pd.isna(df['bb_middle'].iloc[-1]) else None
            
#             # Volatility
#             returns = df['Close'].pct_change()
#             volatility = float(returns.std() * np.sqrt(252) * 100)  # Annualized
            
#             # Support and Resistance
#             recent_data = df.tail(50)
#             support = float(recent_data['Low'].min())
#             resistance = float(recent_data['High'].max())
            
#             # Performance metrics
#             period_return = ((current_price - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
            
#             # Trend analysis
#             trend = self._analyze_trend(df)
            
#             # Technical signals
#             signals = self._generate_technical_signals(
#                 current_price, sma_20, sma_50, sma_200, 
#                 current_rsi, current_macd, current_signal
#             )
            
#             return {
#                 'ticker': ticker_ns,
#                 'current_price': round(current_price, 2),
#                 'price_stats': {
#                     '52_week_high': round(high_52w, 2),
#                     '52_week_low': round(low_52w, 2),
#                     'distance_from_high': round(((current_price - high_52w) / high_52w) * 100, 2),
#                     'distance_from_low': round(((current_price - low_52w) / low_52w) * 100, 2)
#                 },
#                 'volume': {
#                     'current': int(df['Volume'].iloc[-1]),
#                     'average': avg_volume,
#                     'ratio': round(float(df['Volume'].iloc[-1]) / avg_volume, 2)
#                 },
#                 'moving_averages': {
#                     'sma_20': round(sma_20, 2) if sma_20 else None,
#                     'sma_50': round(sma_50, 2) if sma_50 else None,
#                     'sma_200': round(sma_200, 2) if sma_200 else None
#                 },
#                 'indicators': {
#                     'rsi': round(current_rsi, 2),
#                     'rsi_signal': self._get_rsi_signal(current_rsi),
#                     'macd': round(current_macd, 2),
#                     'macd_signal': round(current_signal, 2),
#                     'macd_histogram': round(current_histogram, 2),
#                     'macd_trend': 'Bullish' if current_histogram > 0 else 'Bearish'
#                 },
#                 'bollinger_bands': {
#                     'upper': round(bb_upper, 2) if bb_upper else None,
#                     'middle': round(bb_middle, 2) if bb_middle else None,
#                     'lower': round(bb_lower, 2) if bb_lower else None,
#                     'position': self._get_bb_position(current_price, bb_upper, bb_lower, bb_middle)
#                 },
#                 'support_resistance': {
#                     'support': round(support, 2),
#                     'resistance': round(resistance, 2),
#                     'pivot': round((support + resistance + current_price) / 3, 2)
#                 },
#                 'volatility': round(volatility, 2),
#                 'performance': {
#                     'period_return': round(period_return, 2),
#                     'period': period
#                 },
#                 'trend': trend,
#                 'technical_signals': signals
#             }
        
#         except Exception as e:
#             print(f"Error analyzing {ticker}: {e}")
#             return None
    
#     def _analyze_trend(self, df: pd.DataFrame) -> Dict:
#         """Analyze price trend"""
#         try:
#             recent_data = df.tail(20)
            
#             # Short-term trend (last 5 days)
#             short_term = recent_data.tail(5)['Close']
#             short_trend = 'Up' if short_term.iloc[-1] > short_term.iloc[0] else 'Down'
            
#             # Medium-term trend (last 20 days)
#             medium_trend = 'Up' if df['Close'].iloc[-1] > df['Close'].iloc[-20] else 'Down'
            
#             # Calculate trend strength
#             returns = df['Close'].pct_change().tail(20)
#             positive_days = (returns > 0).sum()
#             trend_strength = (positive_days / 20) * 100
            
#             return {
#                 'short_term': short_trend,
#                 'medium_term': medium_trend,
#                 'strength': round(trend_strength, 2)
#             }
#         except:
#             return {'short_term': 'Neutral', 'medium_term': 'Neutral', 'strength': 50}
    
#     def _get_rsi_signal(self, rsi: float) -> str:
#         """Get RSI signal"""
#         if rsi >= 70:
#             return 'Overbought'
#         elif rsi <= 30:
#             return 'Oversold'
#         else:
#             return 'Neutral'
    
#     def _get_bb_position(self, price, upper, middle, lower) -> str:
#         """Get position relative to Bollinger Bands"""
#         if upper is None or lower is None:
#             return 'Unknown'
        
#         if price >= upper:
#             return 'Above Upper Band (Overbought)'
#         elif price <= lower:
#             return 'Below Lower Band (Oversold)'
#         elif price > middle:
#             return 'Above Middle (Bullish)'
#         else:
#             return 'Below Middle (Bearish)'
    
#     def _generate_technical_signals(self, price, sma_20, sma_50, sma_200, rsi, macd, signal) -> Dict:
#         """Generate buy/sell signals based on technical indicators"""
#         signals = []
#         score = 0
        
#         # Moving Average signals
#         if sma_20 and sma_50:
#             if price > sma_20 > sma_50:
#                 signals.append('Strong uptrend (Price > SMA20 > SMA50)')
#                 score += 2
#             elif price > sma_20:
#                 signals.append('Bullish (Price > SMA20)')
#                 score += 1
#             elif price < sma_20:
#                 signals.append('Bearish (Price < SMA20)')
#                 score -= 1
        
#         # RSI signals
#         if rsi <= 30:
#             signals.append('RSI Oversold - Potential buy opportunity')
#             score += 2
#         elif rsi >= 70:
#             signals.append('RSI Overbought - Consider selling')
#             score -= 2
        
#         # MACD signals
#         if macd > signal:
#             signals.append('MACD Bullish crossover')
#             score += 1
#         else:
#             signals.append('MACD Bearish crossover')
#             score -= 1
        
#         # Overall recommendation
#         if score >= 3:
#             recommendation = 'Strong Buy'
#         elif score >= 1:
#             recommendation = 'Buy'
#         elif score <= -3:
#             recommendation = 'Strong Sell'
#         elif score <= -1:
#             recommendation = 'Sell'
#         else:
#             recommendation = 'Hold'
        
#         return {
#             'signals': signals,
#             'score': score,
#             'recommendation': recommendation
#         }
    
#     def get_market_overview(self) -> Dict:
#         """
#         Get overall market overview
        
#         Returns:
#             Market statistics and sentiment
#         """
#         stocks = self.NIFTY_50_STOCKS[:20]  # Use subset for faster response
        
#         advancing = 0
#         declining = 0
#         unchanged = 0
#         total_volume = 0
        
#         for ticker in stocks:
#             df = self.get_stock_data(ticker, period="2d")
#             if df is not None and len(df) >= 2:
#                 try:
#                     current = df['Close'].iloc[-1]
#                     previous = df['Close'].iloc[-2]
                    
#                     if current > previous:
#                         advancing += 1
#                     elif current < previous:
#                         declining += 1
#                     else:
#                         unchanged += 1
                    
#                     total_volume += int(df['Volume'].iloc[-1])
#                 except:
#                     continue
        
#         total = advancing + declining + unchanged
#         if total > 0:
#             advance_decline_ratio = advancing / declining if declining > 0 else float('inf')
#             market_sentiment = 'Bullish' if advancing > declining else 'Bearish' if declining > advancing else 'Neutral'
#         else:
#             advance_decline_ratio = 0
#             market_sentiment = 'Unknown'
        
#         return {
#             'advancing': advancing,
#             'declining': declining,
#             'unchanged': unchanged,
#             'advance_decline_ratio': round(advance_decline_ratio, 2) if advance_decline_ratio != float('inf') else None,
#             'market_sentiment': market_sentiment,
#             'total_volume': total_volume,
#             'stocks_analyzed': total,
#             'timestamp': datetime.now().isoformat()
#         }
    
#     def compare_stocks(self, tickers: List[str], period: str = "6mo") -> List[Dict]:
#         """
#         Compare multiple stocks side by side
        
#         Args:
#             tickers: List of stock symbols to compare
#             period: Comparison period
            
#         Returns:
#             List of stock comparisons
#         """
#         comparison = []
        
#         for ticker in tickers:
#             df = self.get_stock_data(ticker, period=period)
#             if df is not None and not df.empty:
#                 try:
#                     ticker_ns = self.normalize_ticker(ticker)
#                     current_price = float(df['Close'].iloc[-1])
#                     start_price = float(df['Close'].iloc[0])
                    
#                     returns = ((current_price - start_price) / start_price) * 100
#                     volatility = float(df['Close'].pct_change().std() * np.sqrt(252) * 100)
                    
#                     comparison.append({
#                         'ticker': ticker_ns,
#                         'current_price': round(current_price, 2),
#                         'period_return': round(returns, 2),
#                         'volatility': round(volatility, 2),
#                         'avg_volume': int(df['Volume'].mean()),
#                         'high': round(float(df['High'].max()), 2),
#                         'low': round(float(df['Low'].min()), 2)
#                     })
#                 except Exception as e:
#                     print(f"Error comparing {ticker}: {e}")
#                     continue
        
#         return comparison




# market_analyzer.py
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

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

def _fetch_json(path: str, params: dict = None) -> Optional[dict]:
    s = _nse_session()
    url = _NSE_BASE + path
    try:
        r = s.get(url, params=params or {}, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[nse_fetch] {path} error: {e}")
        return None

def _nse_quote(symbol: str) -> Optional[dict]:
    symbol_clean = symbol.replace('.NS', '').upper()
    return _fetch_json(f"/api/quote-equity?symbol={symbol_clean}")

def _nse_historical(symbol: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
    symbol_clean = symbol.replace('.NS', '').upper()
    path = f"/api/historical/cm/equity?symbol={symbol_clean}&series=[%22EQ%22]&from={from_date}&to={to_date}"
    resp = _fetch_json(path)
    if not resp or 'data' not in resp:
        return None
    rows = resp['data']
    df = pd.DataFrame(rows)
    # normalize date column
    date_col = None
    for cand in ('CH_TIMESTAMP', 'TIMESTAMP', 'DATE'):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        return None
    df.rename(columns={date_col: 'date', 'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'TOTTRDQTY': 'Volume'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])
    df.set_index('date', inplace=True)
    for col in ['Open','High','Low','Close','Volume']:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors='coerce')
    df = df[['Open','High','Low','Close','Volume']].sort_index()
    df = df.dropna()
    return df if not df.empty else None

class MarketAnalyzer:
    NIFTY_50_STOCKS = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
        'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA',
        'TITAN', 'ULTRACEMCO', 'BAJFINANCE', 'NESTLEIND', 'WIPRO',
        'ONGC', 'NTPC', 'POWERGRID', 'M&M', 'TECHM',
        'TATAMOTORS', 'HCLTECH', 'ADANIPORTS', 'COALINDIA', 'TATASTEEL'
    ]

    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)

    def normalize_ticker(self, ticker: str) -> str:
        t = ticker.strip().upper()
        if not t.endswith('.NS') and not t.endswith('.BO'):
            t = f"{t}.NS"
