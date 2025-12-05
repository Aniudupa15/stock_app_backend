import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """
    Stock prediction engine using Random Forest and technical indicators
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        
    def normalize_ticker(self, ticker: str) -> str:
        """
        Add .NS suffix for Indian stocks if not present
        
        Args:
            ticker: Stock symbol (e.g., RELIANCE, TCS)
            
        Returns:
            Normalized ticker with .NS suffix
        """
        ticker = ticker.strip().upper()
        if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
            ticker = f"{ticker}.NS"
        return ticker
    
    def fetch_data(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for historical data
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                print(f"No data returned for {ticker}")
                return None
            
            # Check if we have enough data
            if len(df) < 30:
                print(f"Insufficient data: only {len(df)} rows")
                return None
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            data: Price series
            period: RSI period (default: 14)
            
        Returns:
            RSI values as Series
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill NaN with neutral RSI value
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical indicator features for ML model
        
        Features generated:
        - SMA (5, 10 days)
        - EMA (10 days)
        - RSI (14 days)
        - Volume change
        - Lag returns (1, 2, 3 days)
        - Momentum (5 days)
        - Volatility (10 days)
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            DataFrame with features and target
        """
        data = df.copy()
        
        # Simple Moving Averages
        data['sma_5'] = data['Close'].rolling(window=5).mean()
        data['sma_10'] = data['Close'].rolling(window=10).mean()
        
        # Exponential Moving Average
        data['ema_10'] = data['Close'].ewm(span=10, adjust=False).mean()
        
        # Relative Strength Index
        data['rsi_14'] = self.calculate_rsi(data['Close'], 14)
        
        # Volume change percentage
        data['volume_change'] = data['Volume'].pct_change()
        
        # Price returns (lag features)
        data['return_1d'] = data['Close'].pct_change(1)
        data['return_2d'] = data['Close'].pct_change(2)
        data['return_3d'] = data['Close'].pct_change(3)
        
        # Price momentum (absolute change)
        data['momentum_5'] = data['Close'] - data['Close'].shift(5)
        
        # Volatility (rolling standard deviation)
        data['volatility_10'] = data['Close'].rolling(window=10).std()
        
        # Target variable: Next day's closing price
        data['target'] = data['Close'].shift(-1)
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Replace infinite values with NaN, then drop
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        
        # Cap extreme values (outliers) to prevent numerical issues
        feature_cols = [
            'sma_5', 'sma_10', 'ema_10', 'rsi_14', 'volume_change',
            'return_1d', 'return_2d', 'return_3d', 'momentum_5', 'volatility_10'
        ]
        
        for col in feature_cols:
            if col in data.columns:
                # Cap returns and volume_change to reasonable ranges
                if 'return' in col or 'volume_change' in col:
                    data[col] = data[col].clip(-1, 1)  # Cap at ±100%
                # Cap RSI to valid range
                elif col == 'rsi_14':
                    data[col] = data[col].clip(0, 100)
        
        return data
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
        """
        Train Random Forest Regressor model
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Tuple of (MSE, direction_accuracy)
        """
        # Validate data before training
        if X.isnull().any().any():
            raise ValueError("Training data contains NaN values")
        
        if np.isinf(X.values).any():
            raise ValueError("Training data contains infinite values")
        
        if y.isnull().any():
            raise ValueError("Target data contains NaN values")
        
        if np.isinf(y.values).any():
            raise ValueError("Target data contains infinite values")
        
        # Split data (time-series aware - no shuffle)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Initialize Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        
        # Calculate direction accuracy (up/down prediction)
        actual_direction = np.sign(y_test.values - X_test['sma_10'].values)
        predicted_direction = np.sign(y_pred - X_test['sma_10'].values)
        direction_accuracy = np.mean(actual_direction == predicted_direction)
        
        return mse, direction_accuracy
    
    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """
        Extract feature importance from trained model
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        importance_dict = dict(zip(feature_names, importance.tolist()))
        
        # Sort by importance (descending)
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        # Round to 4 decimal places
        importance_dict = {k: round(v, 4) for k, v in importance_dict.items()}
        
        return importance_dict
    
    def generate_signals(
        self, 
        last_close: float, 
        predicted_close: float
    ) -> Dict[str, any]:
        """
        Generate trading signals based on predicted return
        
        Signal Logic:
        - BUY: Expected return >= +2%
        - HOLD: Expected return between -2% and +2%
        - SELL: Expected return <= -2%
        
        Args:
            last_close: Current closing price
            predicted_close: Predicted next-day closing price
            
        Returns:
            Dictionary with signal, prices, and stop-loss levels
        """
        # Calculate predicted return percentage
        predicted_return = ((predicted_close - last_close) / last_close) * 100
        
        # Generate signal based on threshold
        if predicted_return >= 2.0:
            signal = "BUY"
            entry_price = last_close
            target_price = predicted_close
            stop_loss = last_close * 0.97  # 3% stop loss
            
        elif predicted_return <= -2.0:
            signal = "SELL"
            entry_price = last_close
            target_price = predicted_close
            stop_loss = last_close * 1.03  # 3% stop loss (above entry for short)
            
        else:
            signal = "HOLD"
            entry_price = last_close
            target_price = last_close
            stop_loss = last_close * 0.98  # 2% stop loss
        
        return {
            "signal": signal,
            "predicted_return_pct": round(predicted_return, 2),
            "entry_price": round(entry_price, 2),
            "target_price": round(target_price, 2),
            "stop_loss": round(stop_loss, 2)
        }
    
    def predict(self, ticker: str, period: str = "2y") -> Optional[Dict]:
        """
        Main prediction pipeline
        
        Steps:
        1. Normalize ticker
        2. Fetch historical data
        3. Generate features
        4. Train model
        5. Make prediction
        6. Generate trading signals
        
        Args:
            ticker: Stock symbol
            period: Historical data period
            
        Returns:
            Dictionary with prediction results or None if failed
        """
        try:
            # Step 1: Normalize ticker symbol
            ticker = self.normalize_ticker(ticker)
            print(f"Fetching data for: {ticker}")
            
            # Step 2: Fetch historical data
            df = self.fetch_data(ticker, period)
            if df is None or df.empty:
                print(f"Failed to fetch data for {ticker}")
                return None
            
            print(f"Data fetched: {len(df)} rows")
            
            # Step 3: Generate features
            data = self.generate_features(df)
            
            if len(data) < 30:
                raise ValueError(
                    f"Insufficient data for prediction. Got {len(data)} rows, need at least 30"
                )
            
            print(f"Features generated: {len(data)} rows after cleaning")
            
            # Step 4: Prepare features and target
            feature_cols = [
                'sma_5', 'sma_10', 'ema_10', 'rsi_14', 'volume_change',
                'return_1d', 'return_2d', 'return_3d', 'momentum_5', 'volatility_10'
            ]
            
            X = data[feature_cols]
            y = data['target']
            
            # Final validation: check for any remaining issues
            if X.isnull().any().any():
                null_cols = X.columns[X.isnull().any()].tolist()
                raise ValueError(f"Features contain NaN values in columns: {null_cols}")
            
            if np.isinf(X.values).any():
                inf_cols = X.columns[np.isinf(X.values).any(axis=0)].tolist()
                raise ValueError(f"Features contain infinite values in columns: {inf_cols}")
            
            # Log data statistics for debugging
            print(f"Feature ranges:")
            for col in feature_cols:
                print(f"  {col}: [{X[col].min():.4f}, {X[col].max():.4f}]")
            
            self.feature_names = feature_cols
            
            # Step 5: Train model
            print("Training model...")
            mse, direction_accuracy = self.train_model(X, y)
            print(f"Model trained - MSE: {mse:.2f}, Direction Accuracy: {direction_accuracy*100:.2f}%")
            
            # Step 6: Get last row for prediction
            last_features = X.iloc[-1:].values
            last_close = df['Close'].iloc[-1]
            
            # Step 7: Make prediction
            predicted_close = self.model.predict(last_features)[0]
            print(f"Prediction: {predicted_close:.2f}")
            
            # Step 8: Generate trading signals
            signals = self.generate_signals(last_close, predicted_close)
            
            # Step 9: Get feature importance
            feature_importance = self.get_feature_importance(feature_cols)
            
            # Step 10: Prepare final response
            result = {
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
            
            print(f"Signal: {signals['signal']} ({signals['predicted_return_pct']}%)")
            return result
            
        except Exception as e:
            print(f"Error in prediction pipeline: {e}")
            raise e