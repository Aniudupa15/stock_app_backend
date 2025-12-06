from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import List, Optional

from models import (
    PredictionRequest, 
    PredictionResponse,
    MultiTimeframeAnalysis,
    WatchlistRequest,
    PortfolioAddRequest,
    PortfolioSellRequest,
    ReminderRequest,
    EmailRequest,
    HealthResponse,
    ErrorResponse,
    StockInfo,
    StockAnalysis,
    MarketOverview,
    StockComparison
)
from predictor import StockPredictor
from market_analyzer import MarketAnalyzer
from portfolio_manager import PortfolioManager
# from market_schedule import MarketSchedule # MarketSchedule is missing/commented out
from notification_service import NotificationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Enhanced Indian Stock Market API...")
    logger.info("API is ready to accept requests")
    yield
    # Shutdown
    logger.info("Shutting down API...")

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Indian Stock Market API v3.0",
    description="Advanced stock prediction with Portfolio Management & Notifications",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred", "error_type": type(exc).__name__}
    )

@app.get(
    "/",
    response_model=dict,
    tags=["Root"],
    summary="API Information"
)
async def root():
    """
    Get API information and available endpoints
    """
    return {
        "name": "Enhanced Indian Stock Market API v3.0",
        "version": "3.0.0",
        "description": "Complete stock trading platform with AI, Portfolio & Notifications",
        "new_in_v3": [
            "Portfolio Management (Buy/Sell tracking)",
            "Watchlist with price alerts",
            "Market hours & holiday calendar",
            "Email notifications",
            "Reminders system",
            "Multi-timeframe analysis (Weekly/Monthly/Yearly)",
            "Optimized predictions (auto 2-year period)"
        ],
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /predict": "Next-day price prediction (optimized)",
            "GET /predict/timeframe/{ticker}": "Weekly/Monthly/Yearly analysis",
            "POST /predict/batch": "Batch predictions",
            "GET /market/gainers": "Top gaining stocks",
            "GET /market/losers": "Top losing stocks",
            "GET /market/overview": "Market sentiment",
            "GET /market/status": "Market hours & status",
            "GET /market/holidays": "Upcoming holidays",
            "GET /analysis/{ticker}": "Comprehensive analysis",
            "POST /compare": "Compare stocks",
            "POST /portfolio/add": "Add to portfolio",
            "POST /portfolio/sell": "Sell from portfolio",
            "GET /portfolio": "View portfolio",
            "GET /portfolio/value": "Portfolio valuation",
            "POST /watchlist/add": "Add to watchlist",
            "DELETE /watchlist/{ticker}": "Remove from watchlist",
            "GET /watchlist": "View watchlist",
            "POST /reminders": "Create reminder",
            "GET /reminders": "Get reminders",
            "POST /email/send": "Send custom email",
            "POST /email/portfolio-summary": "Email portfolio summary",
            "GET /docs": "Interactive docs",
            "GET /redoc": "Alternative docs"
        }
    }

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check"
)
async def health_check():
    """
    Check if the API is running and healthy
    """
    return HealthResponse(
        status="healthy",
        message="API is running successfully"
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Get Next-Day Stock Prediction",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "ticker": "RELIANCE.NS",
                        "last_close": 2850.50,
                        "predicted_close": 2920.75,
                        "predicted_return_pct": 2.46,
                        "signal": "BUY",
                        "entry_price": 2850.50,
                        "target_price": 2920.75,
                        "stop_loss": 2765.00,
                        "model_mse": 125.34,
                        "direction_accuracy": 67.50,
                        "feature_importance": {
                            "sma_10": 0.25,
                            "ema_10": 0.18
                        }
                    }
                }
            }
        },
        400: {"description": "Invalid request parameters"},
        404: {"description": "Stock ticker not found"},
        500: {"description": "Internal server error"}
    }
)
async def predict_stock(request: PredictionRequest):
    """
    Predict next-day stock price using optimized ML model (2-year data)
    
    **Parameters:**
    - **ticker**: Stock symbol (e.g., RELIANCE, TCS, INFY)
    
    **Note:** Uses optimal 2-year period for best accuracy
    
    **Returns:**
    - Next-day price prediction
    - Trading signal (BUY/SELL/HOLD)
    - Entry, target, and stop-loss prices
    - Model performance metrics
    - Feature importance analysis
    """
    try:
        logger.info(f"Prediction request for {request.ticker} (using 2y optimal period)")
        
        # Initialize predictor
        predictor = StockPredictor()
        
        # Get prediction
        result = predictor.predict(request.ticker)
        
        # Check if prediction was successful
        if result is None:
            logger.warning(f"No data available for {request.ticker}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unable to fetch data for ticker: {request.ticker}. "
                       f"Please verify the stock symbol is correct and listed on NSE."
            )
        
        logger.info(f"Prediction successful for {request.ticker}: {result['signal']}")
        return PredictionResponse(**result)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction: {str(e)}"
        )

# Optional: Batch prediction endpoint
@app.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
    tags=["Prediction"],
    summary="Batch Stock Predictions"
)
async def predict_batch(tickers: list[str]):
    """
    Get predictions for multiple stocks (uses optimal 2y period)
    
    **Parameters:**
    - **tickers**: List of stock symbols
    
    **Example:**
    ```json
    ["RELIANCE", "TCS", "INFY"]
    ```
    """
    results = []
    predictor = StockPredictor()
    
    for ticker in tickers:
        try:
            result = predictor.predict(ticker)
            if result:
                results.append(PredictionResponse(**result))
        except Exception as e:
            logger.error(f"Error predicting {ticker}: {e}")
            continue
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not generate predictions for any of the provided tickers"
        )
    
    return results

# ==================== MULTI-TIMEFRAME ANALYSIS ====================

@app.get(
    "/predict/timeframe/{ticker}",
    response_model=MultiTimeframeAnalysis,
    tags=["Prediction"],
    summary="Weekly, Monthly & Yearly Analysis"
)
async def get_timeframe_analysis(ticker: str):
    """
    Get comprehensive multi-timeframe analysis
    
    **Includes:**
    - Weekly performance and prediction
    - Monthly performance and prediction
    - Yearly performance and prediction
    
    **Parameters:**
    - **ticker**: Stock symbol (e.g., RELIANCE, TCS)
    
    **Returns:**
    - Historical performance for each timeframe
    - Price predictions for each timeframe
    - Trend analysis
    """
    try:
        logger.info(f"Multi-timeframe analysis for {ticker}")
        predictor = StockPredictor()
        result = predictor.get_multi_timeframe_analysis(ticker)
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unable to fetch data for {ticker}"
            )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in timeframe analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing timeframes: {str(e)}"
        )

# ==================== NEW MARKET ANALYSIS ENDPOINTS ====================

@app.get(
    "/market/gainers",
    response_model=List[StockInfo],
    tags=["Market Analysis"],
    summary="Get Top Gainers"
)
async def get_top_gainers(
    limit: int = Query(default=10, ge=1, le=50, description="Number of top gainers to return")
):
    """
    Get top gaining stocks for the day from NIFTY 50
    
    **Parameters:**
    - **limit**: Number of stocks to return (1-50, default: 10)
    
    **Returns:**
    - List of top gaining stocks with price and volume data
    """
    try:
        logger.info(f"Fetching top {limit} gainers")
        analyzer = MarketAnalyzer()
        gainers = analyzer.get_top_gainers(limit=limit)
        
        if not gainers:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unable to fetch market data. Please try again later."
            )
        
        return gainers
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching gainers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching top gainers: {str(e)}"
        )

@app.get(
    "/market/losers",
    response_model=List[StockInfo],
    tags=["Market Analysis"],
    summary="Get Top Losers"
)
async def get_top_losers(
    limit: int = Query(default=10, ge=1, le=50, description="Number of top losers to return")
):
    """
    Get top losing stocks for the day from NIFTY 50
    
    **Parameters:**
    - **limit**: Number of stocks to return (1-50, default: 10)
    
    **Returns:**
    - List of top losing stocks with price and volume data
    """
    try:
        logger.info(f"Fetching top {limit} losers")
        analyzer = MarketAnalyzer()
        losers = analyzer.get_top_losers(limit=limit)
        
        if not losers:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unable to fetch market data. Please try again later."
            )
        
        return losers
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching losers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching top losers: {str(e)}"
        )

@app.get(
    "/market/overview",
    response_model=MarketOverview,
    tags=["Market Analysis"],
    summary="Get Market Overview"
)
async def get_market_overview():
    """
    Get overall market sentiment and statistics
    
    **Returns:**
    - Number of advancing/declining stocks
    - Advance-Decline ratio
    - Market sentiment (Bullish/Bearish/Neutral)
    - Total trading volume
    """
    try:
        logger.info("Fetching market overview")
        analyzer = MarketAnalyzer()
        overview = analyzer.get_market_overview()
        
        return overview
    
    except Exception as e:
        logger.error(f"Error fetching market overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching market overview: {str(e)}"
        )

@app.get(
    "/analysis/{ticker}",
    response_model=StockAnalysis,
    tags=["Stock Analysis"],
    summary="Get Comprehensive Stock Analysis"
)
async def get_stock_analysis(
    ticker: str,
    period: str = Query(default="1y", description="Analysis period (1mo, 3mo, 6mo, 1y, 2y)")
):
    """
    Get comprehensive technical analysis for a stock
    
    **Includes:**
    - Price statistics (52-week high/low, current price)
    - Moving averages (SMA 20, 50, 200)
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Support and resistance levels
    - Volatility analysis
    - Trend analysis
    - Trading signals and recommendations
    
    **Parameters:**
    - **ticker**: Stock symbol (e.g., RELIANCE, TCS)
    - **period**: Analysis period (default: 1y)
    
    **Example:** `/analysis/RELIANCE?period=6mo`
    """
    try:
        logger.info(f"Analyzing {ticker} for period {period}")
        analyzer = MarketAnalyzer()
        analysis = analyzer.get_stock_analysis(ticker, period)
        
        if analysis is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unable to fetch analysis for {ticker}. Please verify the ticker symbol."
            )
        
        return analysis
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing stock: {str(e)}"
        )

@app.post(
    "/compare",
    response_model=List[StockComparison],
    tags=["Stock Analysis"],
    summary="Compare Multiple Stocks"
)
async def compare_stocks(
    tickers: List[str] = Query(..., description="List of stock tickers to compare"),
    period: str = Query(default="6mo", description="Comparison period")
):
    """
    Compare multiple stocks side by side
    
    **Parameters:**
    - **tickers**: List of stock symbols (e.g., ["RELIANCE", "TCS", "INFY"])
    - **period**: Comparison period (default: 6mo)
    
    **Returns:**
    - Side-by-side comparison of price performance, volatility, and volume
    
    **Example:** `/compare?tickers=RELIANCE&tickers=TCS&tickers=INFY&period=1y`
    """
    try:
        if not tickers or len(tickers) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please provide at least 2 tickers to compare"
            )
        
        if len(tickers) > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 10 stocks can be compared at once"
            )
        
        logger.info(f"Comparing stocks: {', '.join(tickers)}")
        analyzer = MarketAnalyzer()
        comparison = analyzer.compare_stocks(tickers, period)
        
        if not comparison:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unable to fetch data for comparison. Please verify ticker symbols."
            )
        
        return comparison
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing stocks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing stocks: {str(e)}"
        )

# ==================== MARKET SCHEDULE & HOLIDAYS (DISABLED) ====================

@app.get(
    "/market/status",
    tags=["Market Schedule"],
    summary="Get Market Status"
)
async def get_market_status():
    """
    Get current market status and timings
    """
    # Feature disabled due to missing MarketSchedule class
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Market Schedule functionality is currently disabled (MarketSchedule class is missing)"
    )

@app.get(
    "/market/holidays",
    tags=["Market Schedule"],
    summary="Get Upcoming Holidays"
)
async def get_market_holidays(days: int = Query(default=90, ge=1, le=365)):
    """
    Get upcoming market holidays
    """
    # Feature disabled due to missing MarketSchedule class
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Market Schedule functionality is currently disabled (MarketSchedule class is missing)"
    )

# ==================== PORTFOLIO MANAGEMENT ====================

@app.post(
    "/portfolio/add",
    tags=["Portfolio"],
    summary="Add Stock to Portfolio"
)
async def add_to_portfolio(request: PortfolioAddRequest):
    """
    Add stock to your portfolio
    
    **Parameters:**
    - **ticker**: Stock symbol
    - **quantity**: Number of shares
    - **buy_price**: Purchase price per share
    - **buy_date**: Purchase date (optional, defaults to now)
    - **notes**: Optional notes
    
    **Example:**
    ```json
    {
        "ticker": "RELIANCE",
        "quantity": 10,
        "buy_price": 2850.50,
        "notes": "Long term hold"
    }
    ```
    """
    try:
        portfolio_manager = PortfolioManager()
        result = portfolio_manager.add_to_portfolio(
            request.ticker,
            request.quantity,
            request.buy_price,
            request.buy_date,
            request.notes
        )
        return result
    except Exception as e:
        logger.error(f"Error adding to portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding to portfolio: {str(e)}"
        )

@app.post(
    "/portfolio/sell",
    tags=["Portfolio"],
    summary="Sell Stock from Portfolio"
)
async def sell_from_portfolio(request: PortfolioSellRequest):
    """
    Sell stock from your portfolio
    
    **Parameters:**
    - **ticker**: Stock symbol
    - **quantity**: Number of shares to sell
    - **sell_price**: Sale price per share
    - **sell_date**: Sale date (optional, defaults to now)
    - **notes**: Optional notes
    
    **Returns:**
    - Profit/loss calculation
    - Updated portfolio
    """
    try:
        portfolio_manager = PortfolioManager()
        result = portfolio_manager.sell_from_portfolio(
            request.ticker,
            request.quantity,
            request.sell_price,
            request.sell_date,
            request.notes
        )
        return result
    except Exception as e:
        logger.error(f"Error selling from portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error selling from portfolio: {str(e)}"
        )

@app.get(
    "/portfolio",
    tags=["Portfolio"],
    summary="Get Portfolio Holdings"
)
async def get_portfolio():
    """
    Get all portfolio holdings
    
    **Returns:**
    - List of all stocks in portfolio
    - Quantities and average buy prices
    """
    try:
        portfolio_manager = PortfolioManager()
        portfolio = portfolio_manager.get_portfolio()
        return {
            "portfolio": portfolio,
            "total_holdings": len(portfolio)
        }
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting portfolio: {str(e)}"
        )

@app.get(
    "/portfolio/value",
    tags=["Portfolio"],
    summary="Get Portfolio Valuation"
)
async def get_portfolio_value():
    """
    Get current portfolio value and P&L
    
    **Returns:**
    - Total investment
    - Current value
    - Total profit/loss
    - Detailed holdings with individual P&L
    """
    try:
        portfolio_manager = PortfolioManager()
        analyzer = MarketAnalyzer()
        
        # Get portfolio
        portfolio = portfolio_manager.get_portfolio()
        
        # Fetch current prices
        current_prices = {}
        for holding in portfolio:
            ticker = holding['ticker']
            df = analyzer.get_stock_data(ticker, period="1d")
            if df is not None and not df.empty:
                current_prices[ticker] = float(df['Close'].iloc[-1])
        
        # Calculate value
        valuation = portfolio_manager.calculate_portfolio_value(current_prices)
        
        return valuation
    except Exception as e:
        logger.error(f"Error calculating portfolio value: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating portfolio value: {str(e)}"
        )

@app.get(
    "/portfolio/transactions",
    tags=["Portfolio"],
    summary="Get Transaction History"
)
async def get_transactions(
    ticker: Optional[str] = None,
    transaction_type: Optional[str] = None
):
    """
    Get transaction history
    
    **Parameters:**
    - **ticker**: Filter by stock symbol (optional)
    - **transaction_type**: Filter by BUY or SELL (optional)
    
    **Returns:**
    - List of all transactions with dates and prices
    """
    try:
        portfolio_manager = PortfolioManager()
        transactions = portfolio_manager.get_transactions(ticker, transaction_type)
        return {
            "transactions": transactions,
            "total": len(transactions)
        }
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting transactions: {str(e)}"
        )

# ==================== WATCHLIST ====================

@app.post(
    "/watchlist/add",
    tags=["Watchlist"],
    summary="Add Stock to Watchlist"
)
async def add_to_watchlist(request: WatchlistRequest):
    """
    Add stock to watchlist with optional price alert
    
    **Parameters:**
    - **ticker**: Stock symbol
    - **target_price**: Alert price (optional)
    - **notes**: Optional notes
    """
    try:
        portfolio_manager = PortfolioManager()
        result = portfolio_manager.add_to_watchlist(
            request.ticker,
            request.target_price,
            request.notes
        )
        return result
    except Exception as e:
        logger.error(f"Error adding to watchlist: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding to watchlist: {str(e)}"
        )

@app.delete(
    "/watchlist/{ticker}",
    tags=["Watchlist"],
    summary="Remove from Watchlist"
)
async def remove_from_watchlist(ticker: str):
    """Remove stock from watchlist"""
    try:
        portfolio_manager = PortfolioManager()
        result = portfolio_manager.remove_from_watchlist(ticker)
        return result
    except Exception as e:
        logger.error(f"Error removing from watchlist: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing from watchlist: {str(e)}"
        )

@app.get(
    "/watchlist",
    tags=["Watchlist"],
    summary="Get Watchlist"
)
async def get_watchlist():
    """Get all watchlist stocks"""
    try:
        portfolio_manager = PortfolioManager()
        watchlist = portfolio_manager.get_watchlist()
        return {
            "watchlist": watchlist,
            "total": len(watchlist)
        }
    except Exception as e:
        logger.error(f"Error getting watchlist: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting watchlist: {str(e)}"
        )

# ==================== NOTIFICATIONS & REMINDERS ====================

@app.post(
    "/reminders",
    tags=["Notifications"],
    summary="Create Reminder"
)
async def create_reminder(request: ReminderRequest):
    """
    Create a new reminder
    
    **Parameters:**
    - **title**: Reminder title
    - **message**: Reminder message
    - **reminder_time**: When to trigger (ISO format: 2024-12-05T09:30:00)
    - **reminder_type**: Type (GENERAL, EARNINGS, DIVIDEND, etc.)
    - **ticker**: Related stock symbol (optional)
    """
    try:
        notif_service = NotificationService()
        result = notif_service.create_reminder(
            request.title,
            request.message,
            request.reminder_time,
            request.reminder_type,
            request.ticker
        )
        return result
    except Exception as e:
        logger.error(f"Error creating reminder: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating reminder: {str(e)}"
        )

@app.get(
    "/reminders",
    tags=["Notifications"],
    summary="Get Reminders"
)
async def get_reminders(status: Optional[str] = None):
    """
    Get all reminders
    
    **Parameters:**
    - **status**: Filter by ACTIVE or TRIGGERED (optional)
    """
    try:
        notif_service = NotificationService()
        reminders = notif_service.get_reminders(status)
        return {
            "reminders": reminders,
            "total": len(reminders)
        }
    except Exception as e:
        logger.error(f"Error getting reminders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting reminders: {str(e)}"
        )

@app.post(
    "/email/send",
    tags=["Notifications"],
    summary="Send Custom Email"
)
async def send_email(request: EmailRequest):
    """
    Send custom email notification
    
    **Note:** Requires EMAIL_ADDRESS and EMAIL_PASSWORD environment variables
    
    **For Gmail:**
    1. Enable 2FA
    2. Generate App Password: https://support.google.com/accounts/answer/185833
    3. Use App Password as EMAIL_PASSWORD
    """
    try:
        notif_service = NotificationService()
        result = notif_service.send_email(
            request.to_email,
            request.subject,
            request.body,
            request.body_html
        )
        return result
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sending email: {str(e)}"
        )

@app.post(
    "/email/portfolio-summary",
    tags=["Notifications"],
    summary="Email Portfolio Summary"
)
async def email_portfolio_summary(to_email: str):
    """
    Send portfolio summary via email
    
    **Parameters:**
    - **to_email**: Recipient email address
    """
    try:
        portfolio_manager = PortfolioManager()
        analyzer = MarketAnalyzer()
        notif_service = NotificationService()
        
        # Get portfolio and current prices
        portfolio = portfolio_manager.get_portfolio()
        current_prices = {}
        for holding in portfolio:
            ticker = holding['ticker']
            df = analyzer.get_stock_data(ticker, period="1d")
            if df is not None and not df.empty:
                current_prices[ticker] = float(df['Close'].iloc[-1])
        
        # Calculate valuation
        valuation = portfolio_manager.calculate_portfolio_value(current_prices)
        
        # Send email
        result = notif_service.send_portfolio_summary_email(to_email, valuation)
        return result
    except Exception as e:
        logger.error(f"Error sending portfolio summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sending portfolio summary: {str(e)}"
        )

@app.get(
    "/email/config",
    tags=["Notifications"],
    summary="Get Email Configuration Status"
)
async def get_email_config():
    """Check if email notifications are configured"""
    try:
        notif_service = NotificationService()
        return notif_service.get_email_config_status()
    except Exception as e:
        logger.error(f"Error getting email config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting email config: {str(e)}"
        )

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )