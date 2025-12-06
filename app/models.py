from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request model for stock prediction"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., RELIANCE, TCS)")
    
    @validator('ticker')
    def validate_ticker(cls, v):
        if not v or not v.strip():
            raise ValueError("Ticker cannot be empty")
        return v.strip().upper()

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

class MultiTimeframeAnalysis(BaseModel):
    """Multi-timeframe analysis response"""
    ticker: str
    current_price: float
    weekly: Dict
    monthly: Dict
    yearly: Dict

class WatchlistRequest(BaseModel):
    """Request to add stock to watchlist"""
    ticker: str
    target_price: Optional[float] = None
    notes: Optional[str] = None

class PortfolioAddRequest(BaseModel):
    """Request to add stock to portfolio"""
    ticker: str
    quantity: int = Field(..., gt=0)
    buy_price: float = Field(..., gt=0)
    buy_date: Optional[str] = None
    notes: Optional[str] = None

class PortfolioSellRequest(BaseModel):
    """Request to sell stock from portfolio"""
    ticker: str
    quantity: int = Field(..., gt=0)
    sell_price: float = Field(..., gt=0)
    sell_date: Optional[str] = None
    notes: Optional[str] = None

class ReminderRequest(BaseModel):
    """Request to create reminder"""
    title: str
    message: str
    reminder_time: str
    reminder_type: str = "GENERAL"
    ticker: Optional[str] = None

class EmailRequest(BaseModel):
    """Request to send email"""
    to_email: str
    subject: str
    body: str
    body_html: Optional[str] = None

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