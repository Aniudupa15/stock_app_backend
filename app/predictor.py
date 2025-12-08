# import yfinance as yf
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from typing import Optional, Dict, Tuple
# import warnings
# warnings.filterwarnings('ignore')

# class StockPredictor:
#     """
#     Stock prediction engine using Random Forest and technical indicators
#     """
    
#     def __init__(self):
#         self.model = None
#         self.feature_names = []
        
#     def normalize_ticker(self, ticker: str) -> str:
#         """
#         Add .NS suffix for Indian stocks if not present
        
#         Args:
#             ticker: Stock symbol (e.g., RELIANCE, TCS)
            
#         Returns:
#             Normalized ticker with .NS suffix
#         """
#         ticker = ticker.strip().upper()
#         if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
#             ticker = f"{ticker}.NS"
#         return ticker
    
#     def fetch_data(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
#         """
#         Fetch historical stock data from Yahoo Finance
        
#         Args:
#             ticker: Stock ticker symbol
#             period: Time period for historical data
            
#         Returns:
#             DataFrame with OHLCV data or None if failed
#         """
#         try:
#             stock = yf.Ticker(ticker)
#             df = stock.history(period=period)
            
#             if df.empty:
#                 print(f"No data returned for {ticker}")
#                 return None
            
#             # Check if we have enough data
#             if len(df) < 30:
#                 print(f"Insufficient data: only {len(df)} rows")
#                 return None
            
#             return df
            
#         except Exception as e:
#             print(f"Error fetching data for {ticker}: {e}")
#             return None
    
#     def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
#         """
#         Calculate Relative Strength Index (RSI)
        
#         Args:
#             data: Price series
#             period: RSI period (default: 14)
            
#         Returns:
#             RSI values as Series
#         """
#         delta = data.diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
#         # Avoid division by zero
#         rs = gain / loss.replace(0, np.nan)
#         rsi = 100 - (100 / (1 + rs))
        
#         return rsi.fillna(50)  # Fill NaN with neutral RSI value
    
#     def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Generate technical indicator features for ML model
        
#         Features generated:
#         - SMA (5, 10 days)
#         - EMA (10 days)
#         - RSI (14 days)
#         - Volume change
#         - Lag returns (1, 2, 3 days)
#         - Momentum (5 days)
#         - Volatility (10 days)
        
#         Args:
#             df: Raw OHLCV DataFrame
            
#         Returns:
#             DataFrame with features and target
#         """
#         data = df.copy()
        
#         # Simple Moving Averages
#         data['sma_5'] = data['Close'].rolling(window=5).mean()
#         data['sma_10'] = data['Close'].rolling(window=10).mean()
        
#         # Exponential Moving Average
#         data['ema_10'] = data['Close'].ewm(span=10, adjust=False).mean()
        
#         # Relative Strength Index
#         data['rsi_14'] = self.calculate_rsi(data['Close'], 14)
        
#         # Volume change percentage
#         data['volume_change'] = data['Volume'].pct_change()
        
#         # Price returns (lag features)
#         data['return_1d'] = data['Close'].pct_change(1)
#         data['return_2d'] = data['Close'].pct_change(2)
#         data['return_3d'] = data['Close'].pct_change(3)
        
#         # Price momentum (absolute change)
#         data['momentum_5'] = data['Close'] - data['Close'].shift(5)
        
#         # Volatility (rolling standard deviation)
#         data['volatility_10'] = data['Close'].rolling(window=10).std()
        
#         # Target variable: Next day's closing price
#         data['target'] = data['Close'].shift(-1)
        
#         # Drop rows with NaN values
#         data = data.dropna()
        
#         # Replace infinite values with NaN, then drop
#         data = data.replace([np.inf, -np.inf], np.nan)
#         data = data.dropna()
        
#         # Cap extreme values (outliers) to prevent numerical issues
#         feature_cols = [
#             'sma_5', 'sma_10', 'ema_10', 'rsi_14', 'volume_change',
#             'return_1d', 'return_2d', 'return_3d', 'momentum_5', 'volatility_10'
#         ]
        
#         for col in feature_cols:
#             if col in data.columns:
#                 # Cap returns and volume_change to reasonable ranges
#                 if 'return' in col or 'volume_change' in col:
#                     data[col] = data[col].clip(-1, 1)  # Cap at ±100%
#                 # Cap RSI to valid range
#                 elif col == 'rsi_14':
#                     data[col] = data[col].clip(0, 100)
        
#         return data
    
#     def train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
#         """
#         Train Random Forest Regressor model
        
#         Args:
#             X: Feature matrix
#             y: Target variable
            
#         Returns:
#             Tuple of (MSE, direction_accuracy)
#         """
#         # Validate data before training
#         if X.isnull().any().any():
#             raise ValueError("Training data contains NaN values")
        
#         if np.isinf(X.values).any():
#             raise ValueError("Training data contains infinite values")
        
#         if y.isnull().any():
#             raise ValueError("Target data contains NaN values")
        
#         if np.isinf(y.values).any():
#             raise ValueError("Target data contains infinite values")
        
#         # Split data (time-series aware - no shuffle)
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, shuffle=False
#         )
        
#         # Initialize Random Forest model
#         self.model = RandomForestRegressor(
#             n_estimators=100,
#             max_depth=10,
#             min_samples_split=5,
#             min_samples_leaf=2,
#             random_state=42,
#             n_jobs=-1
#         )
        
#         # Train model
#         self.model.fit(X_train, y_train)
        
#         # Predictions on test set
#         y_pred = self.model.predict(X_test)
        
#         # Calculate Mean Squared Error
#         mse = mean_squared_error(y_test, y_pred)
        
#         # Calculate direction accuracy (up/down prediction)
#         actual_direction = np.sign(y_test.values - X_test['sma_10'].values)
#         predicted_direction = np.sign(y_pred - X_test['sma_10'].values)
#         direction_accuracy = np.mean(actual_direction == predicted_direction)
        
#         return mse, direction_accuracy
    
#     def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
#         """
#         Extract feature importance from trained model
        
#         Args:
#             feature_names: List of feature names
            
#         Returns:
#             Dictionary mapping feature names to importance scores
#         """
#         if self.model is None:
#             return {}
        
#         importance = self.model.feature_importances_
#         importance_dict = dict(zip(feature_names, importance.tolist()))
        
#         # Sort by importance (descending)
#         importance_dict = dict(sorted(
#             importance_dict.items(),
#             key=lambda x: x[1],
#             reverse=True
#         ))
        
#         # Round to 4 decimal places
#         importance_dict = {k: round(v, 4) for k, v in importance_dict.items()}
        
#         return importance_dict
    
#     def generate_signals(
#         self, 
#         last_close: float, 
#         predicted_close: float
#     ) -> Dict[str, any]:
#         """
#         Generate trading signals based on predicted return
        
#         Signal Logic:
#         - BUY: Expected return >= +2%
#         - HOLD: Expected return between -2% and +2%
#         - SELL: Expected return <= -2%
        
#         Args:
#             last_close: Current closing price
#             predicted_close: Predicted next-day closing price
            
#         Returns:
#             Dictionary with signal, prices, and stop-loss levels
#         """
#         # Calculate predicted return percentage
#         predicted_return = ((predicted_close - last_close) / last_close) * 100
        
#         # Generate signal based on threshold
#         if predicted_return >= 2.0:
#             signal = "BUY"
#             entry_price = last_close
#             target_price = predicted_close
#             stop_loss = last_close * 0.97  # 3% stop loss
            
#         elif predicted_return <= -2.0:
#             signal = "SELL"
#             entry_price = last_close
#             target_price = predicted_close
#             stop_loss = last_close * 1.03  # 3% stop loss (above entry for short)
            
#         else:
#             signal = "HOLD"
#             entry_price = last_close
#             target_price = last_close
#             stop_loss = last_close * 0.98  # 2% stop loss
        
#         return {
#             "signal": signal,
#             "predicted_return_pct": round(predicted_return, 2),
#             "entry_price": round(entry_price, 2),
#             "target_price": round(target_price, 2),
#             "stop_loss": round(stop_loss, 2)
#         }
    
#     def predict(self, ticker: str) -> Optional[Dict]:
#         """
#         Main prediction pipeline (using optimal 2-year period)
        
#         Steps:
#         1. Normalize ticker
#         2. Fetch historical data (2 years - optimal for accuracy)
#         3. Generate features
#         4. Train model
#         5. Make prediction
#         6. Generate trading signals
        
#         Args:
#             ticker: Stock symbol
            
#         Returns:
#             Dictionary with prediction results or None if failed
#         """
#         try:
#             # Use optimal period for best model performance
#             period = "2y"
            
#             # Step 1: Normalize ticker symbol
#             ticker = self.normalize_ticker(ticker)
#             print(f"Fetching data for: {ticker}")
            
#             # Step 2: Fetch historical data
#             df = self.fetch_data(ticker, period)
#             if df is None or df.empty:
#                 print(f"Failed to fetch data for {ticker}")
#                 return None
            
#             print(f"Data fetched: {len(df)} rows")
            
#             # Step 3: Generate features
#             data = self.generate_features(df)
            
#             if len(data) < 30:
#                 raise ValueError(
#                     f"Insufficient data for prediction. Got {len(data)} rows, need at least 30"
#                 )
            
#             print(f"Features generated: {len(data)} rows after cleaning")
            
#             # Step 4: Prepare features and target
#             feature_cols = [
#                 'sma_5', 'sma_10', 'ema_10', 'rsi_14', 'volume_change',
#                 'return_1d', 'return_2d', 'return_3d', 'momentum_5', 'volatility_10'
#             ]
            
#             X = data[feature_cols]
#             y = data['target']
            
#             # Final validation: check for any remaining issues
#             if X.isnull().any().any():
#                 null_cols = X.columns[X.isnull().any()].tolist()
#                 raise ValueError(f"Features contain NaN values in columns: {null_cols}")
            
#             if np.isinf(X.values).any():
#                 inf_cols = X.columns[np.isinf(X.values).any(axis=0)].tolist()
#                 raise ValueError(f"Features contain infinite values in columns: {inf_cols}")
            
#             # Log data statistics for debugging
#             print(f"Feature ranges:")
#             for col in feature_cols:
#                 print(f"  {col}: [{X[col].min():.4f}, {X[col].max():.4f}]")
            
#             self.feature_names = feature_cols
            
#             # Step 5: Train model
#             print("Training model...")
#             mse, direction_accuracy = self.train_model(X, y)
#             print(f"Model trained - MSE: {mse:.2f}, Direction Accuracy: {direction_accuracy*100:.2f}%")
            
#             # Step 6: Get last row for prediction
#             last_features = X.iloc[-1:].values
#             last_close = df['Close'].iloc[-1]
            
#             # Step 7: Make prediction
#             predicted_close = self.model.predict(last_features)[0]
#             print(f"Prediction: {predicted_close:.2f}")
            
#             # Step 8: Generate trading signals
#             signals = self.generate_signals(last_close, predicted_close)
            
#             # Step 9: Get feature importance
#             feature_importance = self.get_feature_importance(feature_cols)
            
#             # Step 10: Prepare final response
#             result = {
#                 "ticker": ticker,
#                 "last_close": round(last_close, 2),
#                 "predicted_close": round(predicted_close, 2),
#                 "predicted_return_pct": signals["predicted_return_pct"],
#                 "signal": signals["signal"],
#                 "entry_price": signals["entry_price"],
#                 "target_price": signals["target_price"],
#                 "stop_loss": signals["stop_loss"],
#                 "model_mse": round(mse, 2),
#                 "direction_accuracy": round(direction_accuracy * 100, 2),
#                 "feature_importance": feature_importance
#             }
            
#             print(f"Signal: {signals['signal']} ({signals['predicted_return_pct']}%)")
#             return result
            
#         except Exception as e:
#             print(f"Error in prediction pipeline: {e}")
#             raise e
    
#     def get_multi_timeframe_analysis(self, ticker: str) -> Optional[Dict]:
#         """
#         Get weekly, monthly, and yearly predictions and analysis
        
#         Args:
#             ticker: Stock symbol
            
#         Returns:
#             Dictionary with multi-timeframe analysis
#         """
#         try:
#             ticker = self.normalize_ticker(ticker)
            
#             # Fetch longer data for yearly analysis
#             df = self.fetch_data(ticker, period="3y")
#             if df is None or df.empty:
#                 return None
            
#             current_price = df['Close'].iloc[-1]
            
#             # Weekly Analysis (last 7 days)
#             if len(df) >= 7:
#                 week_ago = df['Close'].iloc[-7]
#                 weekly_change = ((current_price - week_ago) / week_ago) * 100
#                 weekly_high = df['High'].tail(7).max()
#                 weekly_low = df['Low'].tail(7).min()
#                 weekly_volume = df['Volume'].tail(7).mean()
#             else:
#                 weekly_change = 0
#                 weekly_high = current_price
#                 weekly_low = current_price
#                 weekly_volume = df['Volume'].mean()
            
#             # Monthly Analysis (last 30 days)
#             if len(df) >= 30:
#                 month_ago = df['Close'].iloc[-30]
#                 monthly_change = ((current_price - month_ago) / month_ago) * 100
#                 monthly_high = df['High'].tail(30).max()
#                 monthly_low = df['Low'].tail(30).min()
#                 monthly_volume = df['Volume'].tail(30).mean()
#             else:
#                 monthly_change = 0
#                 monthly_high = current_price
#                 monthly_low = current_price
#                 monthly_volume = df['Volume'].mean()
            
#             # Yearly Analysis (last 252 trading days ≈ 1 year)
#             if len(df) >= 252:
#                 year_ago = df['Close'].iloc[-252]
#                 yearly_change = ((current_price - year_ago) / year_ago) * 100
#                 yearly_high = df['High'].tail(252).max()
#                 yearly_low = df['Low'].tail(252).min()
#                 yearly_volume = df['Volume'].tail(252).mean()
#             else:
#                 year_ago = df['Close'].iloc[0]
#                 yearly_change = ((current_price - year_ago) / year_ago) * 100
#                 yearly_high = df['High'].max()
#                 yearly_low = df['Low'].min()
#                 yearly_volume = df['Volume'].mean()
            
#             # Calculate predictions for different timeframes
#             weekly_prediction = self._predict_timeframe(df, days=7)
#             monthly_prediction = self._predict_timeframe(df, days=30)
#             yearly_prediction = self._predict_timeframe(df, days=252)
            
#             return {
#                 "ticker": ticker,
#                 "current_price": round(float(current_price), 2),
#                 "weekly": {
#                     "change_pct": round(float(weekly_change), 2),
#                     "high": round(float(weekly_high), 2),
#                     "low": round(float(weekly_low), 2),
#                     "avg_volume": int(weekly_volume),
#                     "predicted_price": round(float(weekly_prediction), 2),
#                     "predicted_change": round(((weekly_prediction - current_price) / current_price) * 100, 2),
#                     "trend": "Bullish" if weekly_change > 0 else "Bearish"
#                 },
#                 "monthly": {
#                     "change_pct": round(float(monthly_change), 2),
#                     "high": round(float(monthly_high), 2),
#                     "low": round(float(monthly_low), 2),
#                     "avg_volume": int(monthly_volume),
#                     "predicted_price": round(float(monthly_prediction), 2),
#                     "predicted_change": round(((monthly_prediction - current_price) / current_price) * 100, 2),
#                     "trend": "Bullish" if monthly_change > 0 else "Bearish"
#                 },
#                 "yearly": {
#                     "change_pct": round(float(yearly_change), 2),
#                     "high": round(float(yearly_high), 2),
#                     "low": round(float(yearly_low), 2),
#                     "avg_volume": int(yearly_volume),
#                     "predicted_price": round(float(yearly_prediction), 2),
#                     "predicted_change": round(((yearly_prediction - current_price) / current_price) * 100, 2),
#                     "trend": "Bullish" if yearly_change > 0 else "Bearish"
#                 }
#             }
        
#         except Exception as e:
#             print(f"Error in multi-timeframe analysis: {e}")
#             return None
    
#     def _predict_timeframe(self, df: pd.DataFrame, days: int) -> float:
#         """
#         Predict price for a specific timeframe
        
#         Args:
#             df: Historical data
#             days: Number of days to predict ahead
            
#         Returns:
#             Predicted price
#         """
#         try:
#             # Use simple moving average and momentum for timeframe prediction
#             if len(df) < days:
#                 return df['Close'].iloc[-1]
            
#             recent_data = df.tail(days)
#             avg_daily_change = recent_data['Close'].pct_change().mean()
#             volatility = recent_data['Close'].pct_change().std()
            
#             # Simple prediction based on trend
#             current_price = df['Close'].iloc[-1]
#             predicted_price = current_price * (1 + avg_daily_change * days)
            
#             # Add some conservative adjustment
#             predicted_price = predicted_price * 0.95 if predicted_price > current_price else predicted_price * 1.05
            
#             return predicted_price
        
#         except:
#             return df['Close'].iloc[-1]



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
        return t

    def get_stock_data(self, ticker: str, period: str = "1d") -> Optional[pd.DataFrame]:
        cache_key = f"nse_{ticker}_{period}"
        now = datetime.utcnow()
        cached = self.cache.get(cache_key)
        if cached and now - cached[0] < self.cache_duration:
            return cached[1].copy()

        # build dates similar to predictor (period in dd-mm-yyyy)
        to_dt = datetime.now()
        if period.endswith('y'):
            years = int(period[:-1]); from_dt = to_dt - timedelta(days=365 * years)
        elif period.endswith('mo'):
            months = int(period[:-2]); from_dt = to_dt - timedelta(days=30 * months)
        elif period.endswith('d'):
            days = int(period[:-1]); from_dt = to_dt - timedelta(days=days)
        else:
            from_dt = to_dt - timedelta(days=30)

        from_str = from_dt.strftime("%d-%m-%Y")
        to_str = to_dt.strftime("%d-%m-%Y")

        df = _nse_historical(ticker, from_str, to_str)
        if df is None:
            return None
        self.cache[cache_key] = (now, df.copy())
        return df

    def get_top_gainers(self, stocks: List[str] = None, limit: int = 10) -> List[Dict]:
        # Try direct NSE API for live gainers first
        resp = _fetch_json("/api/live-analysis-variations?index=gainers")
        results = []
        if resp and isinstance(resp, dict) and 'data' in resp:
            for item in resp['data'][:limit]:
                try:
                    results.append({
                        'ticker': item.get('symbol'),
                        'ticker_ns': self.normalize_ticker(item.get('symbol', '')),
                        'current_price': float(item.get('lastPrice', 0)),
                        'previous_close': float(item.get('previousClose', 0)),
                        'change': round(float(item.get('lastPrice', 0)) - float(item.get('previousClose', 0)), 2),
                        'change_percent': round(float(item.get('pChange', 0)), 2),
                        'volume': int(item.get('tradedQuantity', 0)),
                        'high': float(item.get('dayHigh', 0)),
                        'low': float(item.get('dayLow', 0))
                    })
                except Exception:
                    continue
            return results[:limit]

        # fallback: compute from provided stocks list
        if stocks is None:
            stocks = self.NIFTY_50_STOCKS
        temp = []
        for t in stocks:
            df = self.get_stock_data(t, period="5d")
            if df is not None and len(df) >= 2:
                try:
                    current = float(df['Close'].iloc[-1]); prev = float(df['Close'].iloc[-2])
                    change = current - prev
                    change_pct = (change / prev) * 100 if prev != 0 else 0
                    temp.append({'ticker': t, 'ticker_ns': self.normalize_ticker(t), 'current_price': round(current,2),
                                 'previous_close': round(prev,2), 'change': round(change,2),
                                 'change_percent': round(change_pct,2), 'volume': int(df['Volume'].iloc[-1]),
                                 'high': round(float(df['High'].iloc[-1]),2), 'low': round(float(df['Low'].iloc[-1]),2)})
                except Exception:
                    continue
        temp.sort(key=lambda x: x['change_percent'], reverse=True)
        return temp[:limit]

    def get_top_losers(self, stocks: List[str] = None, limit: int = 10) -> List[Dict]:
        resp = _fetch_json("/api/live-analysis-variations?index=loosers")
        results = []
        if resp and isinstance(resp, dict) and 'data' in resp:
            for item in resp['data'][:limit]:
                try:
                    results.append({
                        'ticker': item.get('symbol'),
                        'ticker_ns': self.normalize_ticker(item.get('symbol', '')),
                        'current_price': float(item.get('lastPrice', 0)),
                        'previous_close': float(item.get('previousClose', 0)),
                        'change': round(float(item.get('lastPrice', 0)) - float(item.get('previousClose', 0)), 2),
                        'change_percent': round(float(item.get('pChange', 0)), 2),
                        'volume': int(item.get('tradedQuantity', 0)),
                        'high': float(item.get('dayHigh', 0)),
                        'low': float(item.get('dayLow', 0))
                    })
                except Exception:
                    continue
            return results[:limit]

        if stocks is None:
            stocks = self.NIFTY_50_STOCKS
        temp = []
        for t in stocks:
            df = self.get_stock_data(t, period="5d")
            if df is not None and len(df) >= 2:
                try:
                    current = float(df['Close'].iloc[-1]); prev = float(df['Close'].iloc[-2])
                    change = current - prev
                    change_pct = (change / prev) * 100 if prev != 0 else 0
                    temp.append({'ticker': t, 'ticker_ns': self.normalize_ticker(t), 'current_price': round(current,2),
                                 'previous_close': round(prev,2), 'change': round(change,2),
                                 'change_percent': round(change_pct,2), 'volume': int(df['Volume'].iloc[-1]),
                                 'high': round(float(df['High'].iloc[-1]),2), 'low': round(float(df['Low'].iloc[-1]),2)})
                except Exception:
                    continue
        temp.sort(key=lambda x: x['change_percent'])
        return temp[:limit]

    def get_market_overview(self) -> Dict:
        """
        Returns a market overview (advancing, declining, unchanged, advance_decline_ratio, sentiment, total_volume)
        """
        resp = _fetch_json("/api/marketStatus")
        if resp and isinstance(resp, dict):
            try:
                # Some endpoints provide marketStatus structure; fallbacks for robustness
                advancing = resp.get('advanceDecline', {}).get('advances', 0) or 0
                declining = resp.get('advanceDecline', {}).get('declines', 0) or 0
                unchanged = resp.get('advanceDecline', {}).get('unchanged', 0) or 0
                total_volume = resp.get('totalVolume', 0) or 0
                a_d_ratio = (advancing / declining) if declining else None
                sentiment = "Neutral"
                if a_d_ratio is not None:
                    if a_d_ratio > 1.05:
                        sentiment = "Bullish"
                    elif a_d_ratio < 0.95:
                        sentiment = "Bearish"
                return {
                    "advancing": int(advancing),
                    "declining": int(declining),
                    "unchanged": int(unchanged),
                    "advance_decline_ratio": round(a_d_ratio, 2) if a_d_ratio is not None else None,
                    "market_sentiment": sentiment,
                    "total_volume": int(total_volume),
                    "stocks_analyzed": len(self.NIFTY_50_STOCKS),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception:
                pass

        # Fallback compute from NIFTY_50 list (slow)
        adv = dec = unc = 0
        total_vol = 0
        for t in self.NIFTY_50_STOCKS:
            df = self.get_stock_data(t, period="1d")
            if df is None or len(df) < 2:
                continue
            try:
                prev = df['Close'].iloc[-2]; cur = df['Close'].iloc[-1]
                if cur > prev: adv += 1
                elif cur < prev: dec += 1
                else: unc += 1
                total_vol += int(df['Volume'].iloc[-1])
            except Exception:
                continue
        a_d_ratio = (adv / dec) if dec else None
        sentiment = "Neutral"
        if a_d_ratio is not None:
            if a_d_ratio > 1.05:
                sentiment = "Bullish"
            elif a_d_ratio < 0.95:
                sentiment = "Bearish"
        return {
            "advancing": adv, "declining": dec, "unchanged": unc,
            "advance_decline_ratio": round(a_d_ratio,2) if a_d_ratio is not None else None,
            "market_sentiment": sentiment,
            "total_volume": total_vol,
            "stocks_analyzed": len(self.NIFTY_50_STOCKS),
            "timestamp": datetime.now().isoformat()
        }

    def get_stock_analysis(self, ticker: str, period: str = "1y") -> Optional[Dict]:
        df = self.get_stock_data(ticker, period=period)
        if df is None or df.empty:
            return None
        try:
            ticker_ns = self.normalize_ticker(ticker)
            current_price = float(df['Close'].iloc[-1])
            high_52w = float(df['High'].max())
            low_52w = float(df['Low'].min())
            avg_volume = int(df['Volume'].mean())

            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            df['sma_200'] = df['Close'].rolling(window=200).mean()

            sma_20 = float(df['sma_20'].iloc[-1]) if not pd.isna(df['sma_20'].iloc[-1]) else None
            sma_50 = float(df['sma_50'].iloc[-1]) if not pd.isna(df['sma_50'].iloc[-1]) else None
            sma_200 = float(df['sma_200'].iloc[-1]) if not pd.isna(df['sma_200'].iloc[-1]) else None

            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50

            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_histogram = macd - signal

            current_macd = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0
            current_signal = float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else 0
            current_histogram = float(macd_histogram.iloc[-1]) if not pd.isna(macd_histogram.iloc[-1]) else 0

            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['Close'].rolling(window=bb_period).mean()
            bb_std_dev = df['Close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)

            bb_upper = float(df['bb_upper'].iloc[-1]) if not pd.isna(df['bb_upper'].iloc[-1]) else None
            bb_lower = float(df['bb_lower'].iloc[-1]) if not pd.isna(df['bb_lower'].iloc[-1]) else None

            volatility = float(df['Close'].pct_change().rolling(window=30).std().iloc[-1]) if not pd.isna(df['Close'].pct_change().rolling(window=30).std().iloc[-1]) else 0

            trend = "Neutral"
            if sma_50 and current_price > sma_50:
                trend = "Up"
            elif sma_50 and current_price < sma_50:
                trend = "Down"

            technical_signals = {
                'rsi': round(current_rsi, 2),
                'macd': round(current_macd, 4),
                'macd_signal': round(current_signal, 4),
                'macd_histogram': round(current_histogram, 4)
            }

            return {
                "ticker": ticker_ns,
                "current_price": round(current_price, 2),
                "price_stats": {
                    "52w_high": round(high_52w, 2),
                    "52w_low": round(low_52w, 2),
                    "avg_volume": avg_volume
                },
                "volume": {"average": avg_volume},
                "moving_averages": {"sma_20": sma_20, "sma_50": sma_50, "sma_200": sma_200},
                "indicators": {"rsi": current_rsi, "macd_trend": "Bullish" if current_macd > current_signal else "Bearish"},
                "bollinger_bands": {"upper": bb_upper, "lower": bb_lower},
                "support_resistance": {},
                "volatility": volatility,
                "performance": {},
                "trend": {"direction": trend},
                "technical_signals": technical_signals,
                "current_price_timestamp": df.index[-1].isoformat()
            }
        except Exception as e:
            print(f"[market_analyzer] Error analyzing {ticker}: {e}")
            return None
