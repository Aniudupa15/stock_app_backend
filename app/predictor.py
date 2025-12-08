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



# predictor.py
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

# NSE helper settings
_NSE_BASE = "https://www.nseindia.com"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com"
}

def _nse_session():
    """Return a requests.Session primed for NSE (homepage GET for cookies)."""
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(_NSE_BASE, timeout=10)
    except Exception:
        # ignore initial failure (we still attempt API requests)
        pass
    return s

def _nse_get_quote(symbol: str) -> Optional[dict]:
    """
    Fetch quote-equity JSON for a single symbol from NSE.
    Returns parsed JSON (dict) or None.
    """
    symbol_clean = symbol.replace('.NS', '').upper()
    url = f"{_NSE_BASE}/api/quote-equity?symbol={symbol_clean}"
    s = _nse_session()
    try:
        r = s.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        # valid responses include 'priceInfo' or 'info'
        if not data:
            return None
        return data
    except Exception as e:
        print(f"[nse_quote] error {symbol_clean}: {e}")
        return None

def _nse_get_historical(symbol: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
    """
    Get historical OHLC for 'symbol' from -> to in dd-mm-yyyy.
    Uses the NSE historical endpoint: /api/historical/cm/equity
    Returns DataFrame with ['Open','High','Low','Close','Volume'] indexed by datetime ascending.
    """
    symbol_clean = symbol.replace('.NS', '').upper()
    url = (f"{_NSE_BASE}/api/historical/cm/equity?symbol={symbol_clean}"
           f"&series=[%22EQ%22]&from={from_date}&to={to_date}")
    s = _nse_session()
    try:
        r = s.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data or 'data' not in data:
            print(f"[nse_hist] no data for {symbol_clean}")
            return None
        rows = data['data']
        df = pd.DataFrame(rows)
        # Expected fields in row: 'CH_TIMESTAMP' or 'TIMESTAMP', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TOTTRDQTY'
        # Normalize keys safely:
        # Try common keys
        date_col = None
        for cand in ('CH_TIMESTAMP', 'TIMESTAMP', 'DATE'):
            if cand in df.columns:
                date_col = cand
                break
        if date_col is None:
            print("[nse_hist] unknown date column")
            return None

        df.rename(columns={
            date_col: 'date',
            'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close',
            'TOTTRDQTY': 'Volume', 'VOLUME': 'Volume', 'DELIV_QTY': 'Volume'
        }, inplace=True)

        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date'])
        df.set_index('date', inplace=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = np.nan

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
        df = df.dropna()
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"[nse_hist] exception {symbol_clean}: {e}")
        return None

class StockPredictor:
    """
    Stock prediction engine using Random Forest and indicators.
    Uses NSE API for historical data (2 years by default).
    """
    def __init__(self):
        self.model = None
        self._cache = {}  # simple in-memory cache: key -> (ts, df)
        self._cache_ttl = timedelta(minutes=10)

    def normalize_ticker(self, ticker: str) -> str:
        t = ticker.strip().upper()
        if not t.endswith('.NS') and not t.endswith('.BO'):
            t = f"{t}.NS"
        return t

    def fetch_data(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """
        Fetch historical data for the ticker. period supports '2y','1y','6mo','3mo','1mo'.
        Returns pandas DataFrame or None.
        """
        ticker_ns = self.normalize_ticker(ticker)
        now = datetime.utcnow()
        cache_key = f"nse_{ticker_ns}_{period}"
        cached = self._cache.get(cache_key)
        if cached:
            ts, df = cached
            if now - ts < self._cache_ttl:
                return df.copy()

        # Compute from/to dates (dd-mm-yyyy)
        to_dt = datetime.now()
        if period.endswith('y'):
            years = int(period[:-1])
            from_dt = to_dt - timedelta(days=365 * years)
        elif period.endswith('mo'):
            months = int(period[:-2])
            from_dt = to_dt - timedelta(days=30 * months)
        elif period.endswith('d'):
            days = int(period[:-1])
            from_dt = to_dt - timedelta(days=days)
        else:
            # default 2y
            from_dt = to_dt - timedelta(days=365 * 2)

        from_str = from_dt.strftime("%d-%m-%Y")
        to_str = to_dt.strftime("%d-%m-%Y")

        # Use NSE historical API
        df = _nse_get_historical(ticker_ns, from_str, to_str)
        if df is None or df.empty:
            return None

        self._cache[cache_key] = (now, df.copy())
        return df

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data['sma_5'] = data['Close'].rolling(window=5).mean()
        data['sma_10'] = data['Close'].rolling(window=10).mean()
        data['ema_10'] = data['Close'].ewm(span=10, adjust=False).mean()
        data['rsi_14'] = self.calculate_rsi(data['Close'], 14)
        data['volume_change'] = data['Volume'].pct_change()
        data['return_1d'] = data['Close'].pct_change(1)
        data['return_2d'] = data['Close'].pct_change(2)
        data['return_3d'] = data['Close'].pct_change(3)
        data['momentum_5'] = data['Close'] - data['Close'].shift(5)
        data['volatility_10'] = data['Close'].rolling(window=10).std()
        data['target'] = data['Close'].shift(-1)
        data = data.dropna()
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        feature_cols = [
            'sma_5', 'sma_10', 'ema_10', 'rsi_14', 'volume_change',
            'return_1d', 'return_2d', 'return_3d', 'momentum_5', 'volatility_10'
        ]
        for col in feature_cols:
            if col in data.columns:
                if 'return' in col or 'volume_change' in col:
                    data[col] = data[col].clip(-1, 1)
                elif col == 'rsi_14':
                    data[col] = data[col].clip(0, 100)
        return data

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
        if X.isnull().any().any():
            raise ValueError("Training data contains NaN values")
        if np.isinf(X.values).any():
            raise ValueError("Training data contains infinite values")
        if y.isnull().any():
            raise ValueError("Target data contains NaN values")
        if np.isinf(y.values).any():
            raise ValueError("Target data contains infinite values")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        actual_direction = np.sign(y_test.values - X_test['sma_10'].values)
        predicted_direction = np.sign(y_pred - X_test['sma_10'].values)
        direction_accuracy = np.mean(actual_direction == predicted_direction)
        return mse, direction_accuracy

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        importance_dict = dict(zip(feature_names, importance.tolist()))
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return {k: round(v, 4) for k, v in importance_dict.items()}

    def generate_signals(self, last_close: float, predicted_close: float) -> Dict[str, any]:
        predicted_return = ((predicted_close - last_close) / last_close) * 100
        if predicted_return >= 2.0:
            signal = "BUY"
            entry_price = last_close
            target_price = predicted_close
            stop_loss = last_close * 0.97
        elif predicted_return <= -2.0:
            signal = "SELL"
            entry_price = last_close
            target_price = predicted_close
            stop_loss = last_close * 1.03
        else:
            signal = "HOLD"
            entry_price = last_close
            target_price = last_close
            stop_loss = last_close * 0.98
        return {
            "signal": signal,
            "predicted_return_pct": round(predicted_return, 2),
            "entry_price": round(entry_price, 2),
            "target_price": round(target_price, 2),
            "stop_loss": round(stop_loss, 2)
        }

    def predict(self, ticker: str) -> Optional[Dict]:
        try:
            ticker_ns = self.normalize_ticker(ticker)
            print(f"Fetching data for: {ticker_ns}")
            df = self.fetch_data(ticker_ns, "2y")
            if df is None or df.empty:
                print(f"[predictor] No data for {ticker_ns}")
                return None

            data = self.generate_features(df)
            if data.empty or 'target' not in data.columns:
                return None

            feature_cols = [
                'sma_5', 'sma_10', 'ema_10', 'rsi_14', 'volume_change',
                'return_1d', 'return_2d', 'return_3d', 'momentum_5', 'volatility_10'
            ]
            X = data[feature_cols]
            y = data['target']
            mse, direction_accuracy = self.train_model(X, y)

            last_row = X.iloc[-1:]
            predicted_close = float(self.model.predict(last_row)[0])
            last_close = float(df['Close'].iloc[-1])
            signals = self.generate_signals(last_close, predicted_close)
            feature_importance = self.get_feature_importance(feature_cols)

            return {
                "ticker": ticker_ns,
                "last_close": round(last_close, 2),
                "predicted_close": round(predicted_close, 2),
                "predicted_return_pct": round(signals['predicted_return_pct'], 2),
                "signal": signals['signal'],
                "entry_price": signals['entry_price'],
                "target_price": signals['target_price'],
                "stop_loss": signals['stop_loss'],
                "model_mse": round(mse, 4),
                "direction_accuracy": round(direction_accuracy * 100, 2),
                "feature_importance": feature_importance
            }
        except Exception as e:
            print(f"[predictor] Prediction error for {ticker}: {e}")
            return None

    def get_multi_timeframe_analysis(self, ticker: str) -> Optional[Dict]:
        """
        Provide weekly / monthly / yearly analysis. Kept minimal to match previous response_model.
        """
        ticker_ns = self.normalize_ticker(ticker)
        df_week = self.fetch_data(ticker_ns, "3mo")
        df_month = self.fetch_data(ticker_ns, "1y")
        df_year = self.fetch_data(ticker_ns, "2y")
        if df_year is None:
            return None
        current_price = float(df_year['Close'].iloc[-1])
        def summary_from_df(df):
            if df is None or df.empty:
                return {}
            return {
                "period_return": round(((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100, 2),
                "volatility": float(df['Close'].pct_change().rolling(window=30).std().dropna().iloc[-1]) if len(df) >= 31 else 0,
                "avg_volume": int(df['Volume'].mean())
            }
        return {
            "ticker": ticker_ns,
            "current_price": round(current_price, 2),
            "weekly": summary_from_df(df_week),
            "monthly": summary_from_df(df_month),
            "yearly": summary_from_df(df_year)
        }
