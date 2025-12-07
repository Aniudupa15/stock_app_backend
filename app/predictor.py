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



# app/stock_predictor.py
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


class StockPredictor:
    """
    Cloud-safe StockPredictor.
    - Uses a requests.Session with User-Agent for yfinance
    - Retry + fallback to Yahoo chart API JSON
    - Returns None on failure (upstream FastAPI will treat as 404)
    """

    def __init__(self):
        self.model = None
        self.feature_names = []
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })
        # short in-memory cache to avoid repeated network calls in same container
        self._cache = {}
        self._cache_ttl = timedelta(minutes=5)

    # ----------------------------
    # Utilities & Cloud-safe fetch
    # ----------------------------
    def normalize_ticker(self, ticker: str) -> str:
        ticker = ticker.strip().upper()
        if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
            ticker = f"{ticker}.NS"
        return ticker

    def _cached(self, key: str):
        entry = self._cache.get(key)
        if not entry:
            return None
        ts, val = entry
        if datetime.now() - ts > self._cache_ttl:
            self._cache.pop(key, None)
            return None
        return val

    def _set_cache(self, key: str, value):
        self._cache[key] = (datetime.now(), value)

    def _fetch_yf_chart_json(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """
        Fallback direct Yahoo chart JSON endpoint (used only if yfinance fails).
        period examples: '2y', '3y', '1y', '5d'
        """
        try:
            # translate period to range param (Yahoo accepts e.g., '2y' as range=2y)
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={period}&interval=1d"
            r = self.session.get(url, timeout=8)
            r.raise_for_status()
            data = r.json()
            result = data.get("chart", {}).get("result")
            if not result:
                return None
            result0 = result[0]
            timestamps = result0.get("timestamp")
            indicators = result0.get("indicators", {}).get("quote", [{}])[0]
            if not timestamps or not indicators:
                return None
            df = pd.DataFrame({
                "Date": pd.to_datetime(timestamps, unit="s"),
                "Open": indicators.get("open"),
                "High": indicators.get("high"),
                "Low": indicators.get("low"),
                "Close": indicators.get("close"),
                "Volume": indicators.get("volume"),
            }).dropna()
            df.set_index("Date", inplace=True)
            if df.empty:
                return None
            return df
        except Exception:
            return None

    def fetch_with_retries(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """
        Robust fetch chain:
        1) yfinance.Ticker(..., session=self.session).history(...)
        2) yf.download(...) with session
        3) Yahoo chart JSON endpoint
        Returns DataFrame or None
        """
        key = f"{ticker}_{period}"
        cached = self._cached(key)
        if cached is not None:
            return cached

        # Attempt 1: yfinance.Ticker with session
        try:
            stock = yf.Ticker(ticker, session=self.session)
            df = stock.history(period=period, auto_adjust=False)
            if df is not None and not df.empty:
                self._set_cache(key, df)
                return df
        except Exception:
            pass

        # Attempt 2: yf.download
        try:
            df = yf.download(tickers=ticker, period=period, progress=False, session=self.session)
            if df is not None and not df.empty:
                self._set_cache(key, df)
                return df
        except Exception:
            pass

        # Attempt 3: direct Yahoo chart JSON
        df = self._fetch_yf_chart_json(ticker, period)
        if df is not None and not df.empty:
            self._set_cache(key, df)
            return df

        return None

    # ----------------------------
    # Core functions (rsi/features/model)
    # ----------------------------
    def fetch_data(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        ticker = self.normalize_ticker(ticker)
        df = self.fetch_with_retries(ticker, period)
        if df is None or df.empty:
            # upstream expects None for 404 handling
            return None
        # Ensure typical columns exist
        expected = {"Open", "High", "Low", "Close", "Volume"}
        if not expected.issubset(set(df.columns)):
            return None
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
        data["sma_5"] = data["Close"].rolling(window=5).mean()
        data["sma_10"] = data["Close"].rolling(window=10).mean()
        data["ema_10"] = data["Close"].ewm(span=10, adjust=False).mean()
        data["rsi_14"] = self.calculate_rsi(data["Close"], 14)
        data["volume_change"] = data["Volume"].pct_change()
        data["return_1d"] = data["Close"].pct_change(1)
        data["return_2d"] = data["Close"].pct_change(2)
        data["return_3d"] = data["Close"].pct_change(3)
        data["momentum_5"] = data["Close"] - data["Close"].shift(5)
        data["volatility_10"] = data["Close"].rolling(window=10).std()
        data["target"] = data["Close"].shift(-1)

        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        # Clip extreme returns
        for col in ["return_1d", "return_2d", "return_3d", "volume_change"]:
            if col in data.columns:
                data[col] = data[col].clip(-1, 1)
        if "rsi_14" in data.columns:
            data["rsi_14"] = data["rsi_14"].clip(0, 100)

        return data

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
        if X.isnull().any().any() or y.isnull().any():
            raise ValueError("Training data contains NaN")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
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
        # direction accuracy vs sma_10
        if "sma_10" in X_test.columns:
            actual_direction = np.sign(y_test.values - X_test["sma_10"].values)
            predicted_direction = np.sign(y_pred - X_test["sma_10"].values)
            direction_accuracy = float(np.mean(actual_direction == predicted_direction))
        else:
            direction_accuracy = 0.0
        return mse, direction_accuracy

    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        d = dict(zip(feature_names, importance.tolist()))
        d = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
        return {k: round(v, 4) for k, v in d.items()}

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

    # ----------------------------
    # Public APIs
    # ----------------------------
    def predict(self, ticker: str) -> Optional[Dict]:
        """
        Returns dict with prediction or None on failure.
        Do not raise — caller expects None to produce 404.
        """
        ticker = self.normalize_ticker(ticker)
        try:
            period = "2y"
            df = self.fetch_data(ticker, period)
            if df is None or df.empty:
                return None

            data = self.generate_features(df)
            if len(data) < 30:
                return None

            feature_cols = [
                "sma_5", "sma_10", "ema_10", "rsi_14", "volume_change",
                "return_1d", "return_2d", "return_3d", "momentum_5", "volatility_10"
            ]
            X = data[feature_cols]
            y = data["target"]

            # final checks
            if X.isnull().any().any() or np.isinf(X.values).any():
                return None

            self.feature_names = feature_cols
            mse, direction_accuracy = self.train_model(X, y)

            last_features = X.iloc[-1:].values
            last_close = float(df["Close"].iloc[-1])
            predicted_close = float(self.model.predict(last_features)[0])
            signals = self.generate_signals(last_close, predicted_close)
            feature_importance = self.get_feature_importance(feature_cols)

            return {
                "ticker": ticker,
                "last_close": round(last_close, 2),
                "predicted_close": round(predicted_close, 2),
                "predicted_return_pct": signals["predicted_return_pct"],
                "signal": signals["signal"],
                "entry_price": signals["entry_price"],
                "target_price": signals["target_price"],
                "stop_loss": signals["stop_loss"],
                "model_mse": round(mse, 2),
                "direction_accuracy": round(direction_accuracy * 100, 2),
                "feature_importance": feature_importance
            }

        except Exception:
            # Never bubble exceptions up to crash the container — return None so caller returns 404
            return None

    def get_multi_timeframe_analysis(self, ticker: str) -> Optional[Dict]:
        ticker = self.normalize_ticker(ticker)
        try:
            df = self.fetch_data(ticker, period="3y")
            if df is None or df.empty:
                return None
            current_price = float(df["Close"].iloc[-1])

            # weekly
            if len(df) >= 7:
                week_ago = df["Close"].iloc[-7]
                weekly_change = ((current_price - week_ago) / week_ago) * 100
                weekly_high = df["High"].tail(7).max()
                weekly_low = df["Low"].tail(7).min()
                weekly_volume = df["Volume"].tail(7).mean()
            else:
                weekly_change = 0
                weekly_high = current_price
                weekly_low = current_price
                weekly_volume = int(df["Volume"].mean())

            # monthly
            if len(df) >= 30:
                month_ago = df["Close"].iloc[-30]
                monthly_change = ((current_price - month_ago) / month_ago) * 100
                monthly_high = df["High"].tail(30).max()
                monthly_low = df["Low"].tail(30).min()
                monthly_volume = df["Volume"].tail(30).mean()
            else:
                monthly_change = 0
                monthly_high = current_price
                monthly_low = current_price
                monthly_volume = int(df["Volume"].mean())

            # yearly
            if len(df) >= 252:
                year_ago = df["Close"].iloc[-252]
                yearly_change = ((current_price - year_ago) / year_ago) * 100
                yearly_high = df["High"].tail(252).max()
                yearly_low = df["Low"].tail(252).min()
                yearly_volume = df["Volume"].tail(252).mean()
            else:
                year_ago = df["Close"].iloc[0]
                yearly_change = ((current_price - year_ago) / year_ago) * 100
                yearly_high = df["High"].max()
                yearly_low = df["Low"].min()
                yearly_volume = df["Volume"].mean()

            # simple timeframe predictions
            def predict_tf(df_local, days):
                if len(df_local) < days:
                    return float(df_local["Close"].iloc[-1])
                recent = df_local.tail(days)
                avg_daily_change = recent["Close"].pct_change().mean()
                current = df_local["Close"].iloc[-1]
                pred = current * (1 + avg_daily_change * days)
                pred = pred * 0.95 if pred > current else pred * 1.05
                return float(pred)

            weekly_prediction = predict_tf(df, 7)
            monthly_prediction = predict_tf(df, 30)
            yearly_prediction = predict_tf(df, 252)

            return {
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "weekly": {
                    "change_pct": round(float(weekly_change), 2),
                    "high": round(float(weekly_high), 2),
                    "low": round(float(weekly_low), 2),
                    "avg_volume": int(weekly_volume),
                    "predicted_price": round(float(weekly_prediction), 2),
                    "predicted_change": round(((weekly_prediction - current_price) / current_price) * 100, 2),
                    "trend": "Bullish" if weekly_change > 0 else "Bearish"
                },
                "monthly": {
                    "change_pct": round(float(monthly_change), 2),
                    "high": round(float(monthly_high), 2),
                    "low": round(float(monthly_low), 2),
                    "avg_volume": int(monthly_volume),
                    "predicted_price": round(float(monthly_prediction), 2),
                    "predicted_change": round(((monthly_prediction - current_price) / current_price) * 100, 2),
                    "trend": "Bullish" if monthly_change > 0 else "Bearish"
                },
                "yearly": {
                    "change_pct": round(float(yearly_change), 2),
                    "high": round(float(yearly_high), 2),
                    "low": round(float(yearly_low), 2),
                    "avg_volume": int(yearly_volume),
                    "predicted_price": round(float(yearly_prediction), 2),
                    "predicted_change": round(((yearly_prediction - current_price) / current_price) * 100, 2),
                    "trend": "Bullish" if yearly_change > 0 else "Bearish"
                }
            }
        except Exception:
            return None
