from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List

class PredictionRequest(BaseModel):
    """Request model for stock prediction"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., RELIANCE, TCS)")
    period: str = Field(default="2y", description="Historical data period (1mo, 3mo, 6mo, 1y, 2y, 5y)")
    
    @validator('ticker')
    def validate_ticker(cls, v):
        if not v or not v.strip():
            raise ValueError("Ticker cannot be empty")
        return v.strip().upper()
    
    @validator('period')
    def validate_period(cls, v):
        valid_periods = ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']
        if v not in valid_periods:
            raise ValueError(f"Period must be one of {valid_periods}")
        return v

class CompareRequest(BaseModel):
    """Request model for stock comparison"""
    tickers: List[str] = Field(..., min_items=2, max_items=10, description="List of stock tickers to compare")
    period: str = Field(default="6mo", description="Comparison period")
    
    @validator('tickers')
    def validate_tickers(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Please provide at least 2 tickers")
        if len(v) > 10:
            raise ValueError("Maximum 10 tickers allowed")
        return [ticker.strip().upper() for ticker in v]
    
    @validator('period')
    def validate_period(cls, v):
        valid_periods = ['1mo', '3mo', '6mo', '1y', '2y', '5y']
        if v not in valid_periods:
            raise ValueError(f"Period must be one of {valid_periods}")
        return v

class PredictionResponse(BaseModel):
    """Response model for stock prediction"""
    ticker: str
    last_close: float
    predicted_close: float
    predicted_return_pct: float
    signal: str
    entry_price: float
    target_price: float
    stop_loss: float
    model_mse: float
    direction_accuracy: float
    feature_importance: Dict[str, float]

class StockInfo(BaseModel):
    """Stock information model"""
    ticker: str
    ticker_ns: str
    current_price: float
    previous_close: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float

class StockAnalysis(BaseModel):
    """Comprehensive stock analysis model"""
    ticker: str
    current_price: float
    price_stats: Dict
    volume: Dict
    moving_averages: Dict
    indicators: Dict
    bollinger_bands: Dict
    support_resistance: Dict
    volatility: float
    performance: Dict
    trend: Dict
    technical_signals: Dict

class MarketOverview(BaseModel):
    """Market overview model"""
    advancing: int
    declining: int
    unchanged: int
    advance_decline_ratio: Optional[float]
    market_sentiment: str
    total_volume: int
    stocks_analyzed: int
    timestamp: str

class StockComparison(BaseModel):
    """Stock comparison model"""
    ticker: str
    current_price: float
    period_return: float
    volatility: float
    avg_volume: int
    high: float
    low: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
    error_type: Optional[str] = None