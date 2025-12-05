from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import List

from models import (
    PredictionRequest, 
    PredictionResponse, 
    HealthResponse,
    ErrorResponse,
    StockInfo,
    StockAnalysis,
    MarketOverview,
    StockComparison,
    CompareRequest
)
from predictor import StockPredictor
from market_analyzer import MarketAnalyzer

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
    title="Enhanced Indian Stock Market API",
    description="Advanced ML-powered stock prediction and analysis API for NSE stocks",
    version="2.0.0",
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
        "name": "Enhanced Indian Stock Market API",
        "version": "2.0.0",
        "description": "Advanced ML-powered prediction and analysis API for NSE stocks",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /predict": "Stock price prediction with ML",
            "POST /predict/batch": "Batch predictions for multiple stocks",
            "GET /market/gainers": "Top gaining stocks",
            "GET /market/losers": "Top losing stocks",
            "GET /market/overview": "Overall market sentiment",
            "GET /analysis/{ticker}": "Comprehensive stock analysis",
            "POST /compare": "Compare multiple stocks",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "Alternative API documentation"
        },
        "features": [
            "ML-based price prediction",
            "Technical analysis with 10+ indicators",
            "Top gainers/losers tracking",
            "Market sentiment analysis",
            "Stock comparison tools",
            "Real-time data from Yahoo Finance"
        ]
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
    summary="Get Stock Prediction",
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
    Predict next-day stock price and generate trading signals
    
    **Parameters:**
    - **ticker**: Stock symbol (e.g., RELIANCE, TCS, INFY)
    - **period**: Historical data period (1mo, 3mo, 6mo, 1y, 2y, 5y)
    
    **Returns:**
    - Stock prediction with ML insights
    - Trading signal (BUY/SELL/HOLD)
    - Entry, target, and stop-loss prices
    - Model performance metrics
    - Feature importance analysis
    
    **Example Indian stocks:**
    - RELIANCE (Reliance Industries)
    - TCS (Tata Consultancy Services)
    - INFY (Infosys)
    - HDFCBANK (HDFC Bank)
    - TATAMOTORS (Tata Motors)
    """
    try:
        logger.info(f"Prediction request for {request.ticker} with period {request.period}")
        
        # Initialize predictor
        predictor = StockPredictor()
        
        # Get prediction
        result = predictor.predict(request.ticker, request.period)
        
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
async def predict_batch(tickers: list[str], period: str = "2y"):
    """
    Get predictions for multiple stocks at once
    
    **Parameters:**
    - **tickers**: List of stock symbols
    - **period**: Historical data period (default: 2y)
    
    **Example:**
    ```json
    {
        "tickers": ["RELIANCE", "TCS", "INFY"],
        "period": "2y"
    }
    ```
    """
    results = []
    predictor = StockPredictor()
    
    for ticker in tickers:
        try:
            result = predictor.predict(ticker, period)
            if result:
                results.append(PredictionResponse(**result))
        except Exception as e:
            logger.error(f"Error predicting {ticker}: {e}")
            # Continue with other tickers
            continue
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not generate predictions for any of the provided tickers"
        )
    
    return results

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
async def compare_stocks(request: CompareRequest):
    """
    Compare multiple stocks side by side
    
    **Request Body:**
    ```json
    {
        "tickers": ["RELIANCE", "TCS", "INFY"],
        "period": "6mo"
    }
    ```
    
    **Parameters:**
    - **tickers**: List of stock symbols (minimum 2, maximum 10)
    - **period**: Comparison period (default: 6mo)
    
    **Returns:**
    - Side-by-side comparison of price performance, volatility, and volume
    """
    try:
        logger.info(f"Comparing stocks: {', '.join(request.tickers)}")
        analyzer = MarketAnalyzer()
        comparison = analyzer.compare_stocks(request.tickers, request.period)
        
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

# Alternative GET endpoint for URL-based comparison
@app.get(
    "/compare/get",
    response_model=List[StockComparison],
    tags=["Stock Analysis"],
    summary="Compare Stocks (GET method)"
)
async def compare_stocks_get(
    tickers: List[str] = Query(..., description="Stock tickers to compare"),
    period: str = Query(default="6mo", description="Comparison period")
):
    """
    Compare multiple stocks using GET method
    
    **Example:** `/compare/get?tickers=RELIANCE&tickers=TCS&tickers=INFY&period=6mo`
    
    **Parameters:**
    - **tickers**: Stock symbols (repeat parameter for each ticker)
    - **period**: Comparison period
    """
    request = CompareRequest(tickers=tickers, period=period)
    return await compare_stocks(request)

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )