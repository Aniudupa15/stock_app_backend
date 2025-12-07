# Indian Stock Market Prediction & Portfolio Management System

## 📖 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
  - [Health & Status](#health--status)
  - [Stock Predictions](#stock-predictions)
  - [Market Data](#market-data)
  - [Stock Analysis](#stock-analysis)
  - [Market Schedule](#market-schedule)
  - [Portfolio Management](#portfolio-management)
  - [Watchlist](#watchlist)
  - [Notifications & Reminders](#notifications--reminders)
- [Data Models](#data-models)
- [Examples](#examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This is a comprehensive **FastAPI-based backend system** for Indian stock market analysis and portfolio management. It combines:

- **Machine Learning** for price predictions using Random Forest algorithm
- **Technical Analysis** with 15+ indicators (RSI, MACD, Bollinger Bands, etc.)
- **Portfolio Management** with real-time P&L tracking
- **Market Intelligence** with top gainers/losers and sentiment analysis
- **Automation** with email notifications and reminders
- **Multi-Timeframe Analysis** for weekly, monthly, and yearly insights

### Key Technologies
- **Framework**: FastAPI (Python 3.8+)
- **ML Engine**: scikit-learn (Random Forest Regressor)
- **Data Source**: Yahoo Finance (yfinance)
- **Data Processing**: pandas, numpy
- **Market Data**: NSE (National Stock Exchange) stocks
- **Storage**: JSON files (portfolio, watchlist, transactions)
- **Notifications**: SMTP email integration

---

## ✨ Features

### 🤖 Machine Learning
- **Optimized Predictions**: Automatic 2-year historical data for best accuracy
- **10 Technical Indicators**: SMA, EMA, RSI, volume, momentum, volatility
- **Trading Signals**: Intelligent BUY/SELL/HOLD recommendations
- **Risk Management**: Auto-calculated entry, target, and stop-loss prices
- **Model Metrics**: MSE, direction accuracy, feature importance

### 📊 Market Intelligence
- **Top Movers**: Real-time gainers and losers from NIFTY 50
- **Market Sentiment**: Advance-Decline ratio and overall sentiment
- **Comprehensive Analysis**: 15+ indicators per stock
- **Stock Comparison**: Side-by-side analysis of multiple stocks
- **Multi-Timeframe**: Weekly, monthly, yearly predictions

### 💼 Portfolio Management
- **Buy/Sell Tracking**: Complete transaction history
- **Real-time P&L**: Live profit/loss calculations
- **Position Management**: Average price calculation for multiple buys
- **Performance Analytics**: Per-stock and overall portfolio metrics
- **Transaction Log**: Detailed buy/sell records with timestamps

### 👀 Watchlist
- **Stock Tracking**: Monitor stocks of interest
- **Price Alerts**: Set target prices for notifications
- **Custom Notes**: Add personal research notes
- **Easy Management**: Add/remove stocks instantly

### 📅 Market Schedule
- **Live Status**: Real-time market open/closed status
- **Trading Hours**: Pre-market, regular, post-market timings
- **Holiday Calendar**: NSE holidays for 2024-2025
- **Countdown Timers**: Time until market open/close

### 📧 Notifications
- **Email Alerts**: Portfolio summaries and custom messages
- **Reminders**: Set date-based reminders for earnings, dividends
- **HTML Templates**: Professional email formatting
- **Gmail Integration**: Easy setup with App Passwords

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                        │
│                     (Port 8000, CORS Enabled)               │
└──────────────────┬──────────────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌────────────┐
│Predictor│  │ Market   │  │  Portfolio │
│ Engine  │  │ Analyzer │  │  Manager   │
└────┬────┘  └────┬─────┘  └─────┬──────┘
     │            │               │
     ▼            ▼               ▼
┌─────────────────────────────────────┐
│        Yahoo Finance API            │
│    (Historical & Real-time Data)    │
└─────────────────────────────────────┘
     │            │               │
     ▼            ▼               ▼
┌─────────────────────────────────────┐
│       Data Storage (JSON)           │
│  • portfolio.json                   │
│  • watchlist.json                   │
│  • transactions.json                │
│  • reminders.json                   │
│  • notifications.json               │
└─────────────────────────────────────┘
```

### Components

**1. Predictor Engine (`predictor.py`)**
- Fetches historical data
- Generates technical features
- Trains Random Forest model
- Makes predictions
- Calculates trading signals

**2. Market Analyzer (`market_analyzer.py`)**
- Tracks market movers
- Calculates market sentiment
- Performs technical analysis
- Compares stocks

**3. Portfolio Manager (`portfolio_manager.py`)**
- Manages stock holdings
- Tracks transactions
- Calculates P&L
- Maintains watchlist

**4. Market Schedule (`market_schedule.py`)**
- Checks market status
- Manages holiday calendar
- Calculates trading hours

**5. Notification Service (`notification_service.py`)**
- Sends emails
- Manages reminders
- Logs notifications

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection (for stock data)
- Gmail account (optional, for notifications)

### Step 1: Clone/Create Project

```bash
mkdir stock-prediction-backend
cd stock-prediction-backend
mkdir app data
```

### Step 2: Create Files

Create the following file structure:
```
stock-prediction-backend/
├── app/
│   ├── __init__.py                  # Empty file
│   ├── main.py                      # FastAPI application
│   ├── models.py                    # Pydantic models
│   ├── predictor.py                 # ML prediction engine
│   ├── market_analyzer.py           # Market analysis
│   ├── portfolio_manager.py         # Portfolio management
│   ├── market_schedule.py           # Market hours & holidays
│   └── notification_service.py      # Email & reminders
├── data/                            # Auto-created storage
├── requirements.txt
├── .env
└── README.md
```

### Step 3: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

**requirements.txt:**
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
yfinance==0.2.32
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
python-multipart==0.0.6
python-dotenv==1.0.0
requests==2.31.0
pytz==2023.3
```

### Step 4: Configure Environment

Create `.env` file:
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG_MODE=True

# CORS Settings
ALLOWED_ORIGINS=*

# Model Configuration
MODEL_N_ESTIMATORS=100
MODEL_MAX_DEPTH=10
MODEL_MIN_SAMPLES_SPLIT=5

# Data Configuration
DEFAULT_PERIOD=2y
MIN_DATA_ROWS=30

# Email Configuration (Optional)
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

### Step 5: Run Server

```bash
uvicorn app.main:app --reload
```

Server will start at: **http://localhost:8000**

---

## ⚡ Quick Start

### 1. Check Server Health
```bash
curl http://localhost:8000/health
```

### 2. Get Market Status
```bash
curl http://localhost:8000/market/status
```

### 3. Predict a Stock
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker":"RELIANCE"}'
```

### 4. Add to Portfolio
```bash
curl -X POST http://localhost:8000/portfolio/add \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "RELIANCE",
    "quantity": 10,
    "buy_price": 2850.50
  }'
```

### 5. View Interactive Docs
Open in browser: **http://localhost:8000/docs**

---

## 📚 API Endpoints

### Health & Status

#### `GET /` - API Information
Get API details and available endpoints.

**Parameters:** None

**Response:**
```json
{
  "name": "Enhanced Indian Stock Market API v3.0",
  "version": "3.0.0",
  "description": "Complete stock trading platform",
  "new_in_v3": [...],
  "endpoints": {...}
}
```

**Example:**
```bash
curl http://localhost:8000/
```

---

#### `GET /health` - Health Check
Check if API is running.

**Parameters:** None

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running successfully"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### Stock Predictions

#### `POST /predict` - Next-Day Price Prediction

Predict next-day closing price using ML model (optimized with 2-year data).

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| ticker | string | Yes | Stock symbol (e.g., RELIANCE, TCS) |

**Request Body:**
```json
{
  "ticker": "RELIANCE"
}
```

**Response:**
```json
{
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
    "ema_10": 0.18,
    "rsi_14": 0.15,
    "return_1d": 0.12,
    "volatility_10": 0.10,
    "momentum_5": 0.08,
    "sma_5": 0.06,
    "return_2d": 0.03,
    "volume_change": 0.02,
    "return_3d": 0.01
  }
}
```

**Signals:**
- **BUY**: Expected return ≥ +2%
- **HOLD**: Expected return between -2% and +2%
- **SELL**: Expected return ≤ -2%

**Examples:**
```bash
# cURL
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker":"RELIANCE"}'

# Python
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={"ticker": "RELIANCE"}
)
print(response.json())

# JavaScript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({ticker: 'RELIANCE'})
})
.then(r => r.json())
.then(data => console.log(data));
```

---

#### `POST /predict/batch` - Batch Predictions

Get predictions for multiple stocks at once.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| tickers | array | Yes | List of stock symbols |

**Request Body:**
```json
["RELIANCE", "TCS", "INFY"]
```

**Response:** Array of prediction objects (same as single prediction)

**Example:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '["RELIANCE", "TCS", "INFY"]'
```

---

#### `GET /predict/timeframe/{ticker}` - Multi-Timeframe Analysis

Get weekly, monthly, and yearly analysis and predictions.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| ticker | string | Yes | Stock symbol (path parameter) |

**Response:**
```json
{
  "ticker": "RELIANCE.NS",
  "current_price": 2850.50,
  "weekly": {
    "change_pct": 2.5,
    "high": 2900.00,
    "low": 2750.00,
    "avg_volume": 5200000,
    "predicted_price": 2900.00,
    "predicted_change": 1.74,
    "trend": "Bullish"
  },
  "monthly": {
    "change_pct": 5.2,
    "high": 2950.00,
    "low": 2700.00,
    "avg_volume": 4800000,
    "predicted_price": 2980.00,
    "predicted_change": 4.54,
    "trend": "Bullish"
  },
  "yearly": {
    "change_pct": 15.8,
    "high": 3100.00,
    "low": 2200.00,
    "avg_volume": 4500000,
    "predicted_price": 3200.00,
    "predicted_change": 12.26,
    "trend": "Bullish"
  }
}
```

**Example:**
```bash
curl http://localhost:8000/predict/timeframe/RELIANCE
```

---

### Market Data

#### `GET /market/gainers` - Top Gaining Stocks

Get top performing stocks from NIFTY 50.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| limit | integer | No | 10 | Number of stocks (1-50) |

**Response:**
```json
[
  {
    "ticker": "RELIANCE",
    "ticker_ns": "RELIANCE.NS",
    "current_price": 2850.50,
    "previous_close": 2780.00,
    "change": 70.50,
    "change_percent": 2.54,
    "volume": 5234567,
    "high": 2875.00,
    "low": 2800.00
  },
  ...
]
```

**Example:**
```bash
# Top 10 gainers
curl http://localhost:8000/market/gainers?limit=10

# Top 5 gainers
curl http://localhost:8000/market/gainers?limit=5
```

---

#### `GET /market/losers` - Top Losing Stocks

Get worst performing stocks from NIFTY 50.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| limit | integer | No | 10 | Number of stocks (1-50) |

**Response:** Same format as gainers (with negative change_percent)

**Example:**
```bash
curl http://localhost:8000/market/losers?limit=10
```

---

#### `GET /market/overview` - Market Overview

Get overall market sentiment and statistics.

**Parameters:** None

**Response:**
```json
{
  "advancing": 15,
  "declining": 8,
  "unchanged": 2,
  "advance_decline_ratio": 1.88,
  "market_sentiment": "Bullish",
  "total_volume": 125000000,
  "stocks_analyzed": 25,
  "timestamp": "2024-12-04T14:30:00"
}
```

**Market Sentiment:**
- **Bullish**: More advancing than declining
- **Bearish**: More declining than advancing
- **Neutral**: Equal advancing and declining

**Example:**
```bash
curl http://localhost:8000/market/overview
```

---

### Stock Analysis

#### `GET /analysis/{ticker}` - Comprehensive Stock Analysis

Get detailed technical analysis with 15+ indicators.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| ticker | string | Yes | - | Stock symbol (path parameter) |
| period | string | No | 1y | Analysis period (1mo, 3mo, 6mo, 1y, 2y) |

**Response:**
```json
{
  "ticker": "RELIANCE.NS",
  "current_price": 2850.50,
  "price_stats": {
    "52_week_high": 3100.00,
    "52_week_low": 2200.00,
    "distance_from_high": -8.05,
    "distance_from_low": 29.57
  },
  "volume": {
    "current": 5234567,
    "average": 4500000,
    "ratio": 1.16
  },
  "moving_averages": {
    "sma_20": 2820.30,
    "sma_50": 2750.80,
    "sma_200": 2650.50
  },
  "indicators": {
    "rsi": 62.50,
    "rsi_signal": "Neutral",
    "macd": 25.30,
    "macd_signal": 22.10,
    "macd_histogram": 3.20,
    "macd_trend": "Bullish"
  },
  "bollinger_bands": {
    "upper": 2950.00,
    "middle": 2820.00,
    "lower": 2690.00,
    "position": "Above Middle (Bullish)"
  },
  "support_resistance": {
    "support": 2750.00,
    "resistance": 2950.00,
    "pivot": 2850.17
  },
  "volatility": 18.45,
  "performance": {
    "period_return": 15.67,
    "period": "1y"
  },
  "trend": {
    "short_term": "Up",
    "medium_term": "Up",
    "strength": 65.00
  },
  "technical_signals": {
    "signals": [
      "Strong uptrend (Price > SMA20 > SMA50)",
      "RSI Neutral",
      "MACD Bullish crossover"
    ],
    "score": 4,
    "recommendation": "Strong Buy"
  }
}
```

**Recommendations:**
- **Strong Buy**: Score ≥ 3
- **Buy**: Score = 1 or 2
- **Hold**: Score = 0
- **Sell**: Score = -1 or -2
- **Strong Sell**: Score ≤ -3

**Example:**
```bash
# 1 year analysis
curl http://localhost:8000/analysis/RELIANCE?period=1y

# 6 month analysis
curl http://localhost:8000/analysis/TCS?period=6mo
```

---

#### `GET /compare` - Compare Multiple Stocks

Compare up to 10 stocks side-by-side.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| tickers | array | Yes | - | Stock symbols (query params) |
| period | string | No | 6mo | Comparison period |

**Response:**
```json
[
  {
    "ticker": "RELIANCE.NS",
    "current_price": 2850.50,
    "period_return": 12.50,
    "volatility": 18.45,
    "avg_volume": 4500000,
    "high": 3100.00,
    "low": 2400.00
  },
  {
    "ticker": "TCS.NS",
    "current_price": 3650.75,
    "period_return": 8.30,
    "volatility": 15.20,
    "avg_volume": 3200000,
    "high": 3800.00,
    "low": 3300.00
  }
]
```

**Example:**
```bash
# Compare 3 stocks
curl "http://localhost:8000/compare?tickers=RELIANCE&tickers=TCS&tickers=INFY&period=6mo"

# Compare with 1 year data
curl "http://localhost:8000/compare?tickers=HDFCBANK&tickers=ICICIBANK&period=1y"
```

---

### Market Schedule

#### `GET /market/status` - Market Status

Get real-time market status and timings.

**Parameters:** None

**Response:**
```json
{
  "is_open": true,
  "status": "OPEN",
  "session": "MARKET_HOURS",
  "current_time": "2024-12-04 14:30:00 IST",
  "market_open_time": "09:15",
  "market_close_time": "15:30",
  "next_market_day": "2024-12-04",
  "timings": {
    "regular_market": {
      "open": "09:15",
      "close": "15:30",
      "duration": "6 hours 15 minutes"
    },
    "pre_market": {
      "open": "09:00",
      "close": "09:15",
      "duration": "15 minutes"
    },
    "post_market": {
      "open": "15:40",
      "close": "16:00",
      "duration": "20 minutes"
    },
    "timezone": "IST (Asia/Kolkata)",
    "trading_days": "Monday to Friday"
  },
  "countdown": {
    "status": "MARKET_OPEN",
    "hours_until_close": 1,
    "minutes_until_close": 0,
    "time_until_close": "1h 0m"
  }
}
```

**Status Values:**
- **OPEN**: Regular market hours
- **CLOSED**: Market closed
- **PRE_MARKET_OPEN**: Pre-market session (9:00-9:15)
- **POST_MARKET_OPEN**: Post-market session (15:40-16:00)

**Session Values:**
- **BEFORE_MARKET**: Before 9:00 AM
- **PRE_MARKET**: 9:00-9:15 AM
- **MARKET_HOURS**: 9:15 AM-3:30 PM
- **POST_MARKET**: 3:40-4:00 PM
- **AFTER_MARKET**: After 4:00 PM

**Example:**
```bash
curl http://localhost:8000/market/status
```

---

#### `GET /market/holidays` - Holiday Calendar

Get upcoming NSE holidays.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| days | integer | No | 90 | Look ahead N days (max 365) |

**Response:**
```json
{
  "upcoming_holidays": [
    {
      "date": "2024-12-25",
      "day": "Wednesday",
      "days_from_now": 21
    },
    {
      "date": "2025-01-26",
      "day": "Sunday",
      "days_from_now": 53
    }
  ],
  "total_holidays": 2,
  "days_checked": 90
}
```

**Example:**
```bash
# Next 90 days
curl http://localhost:8000/market/holidays?days=90

# Next 180 days
curl http://localhost:8000/market/holidays?days=180
```

---

### Portfolio Management

#### `POST /portfolio/add` - Add Stock to Portfolio

Add a stock purchase to your portfolio.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| ticker | string | Yes | Stock symbol |
| quantity | integer | Yes | Number of shares (> 0) |
| buy_price | float | Yes | Purchase price per share (> 0) |
| buy_date | string | No | Purchase date (ISO format) |
| notes | string | No | Optional notes |

**Request Body:**
```json
{
  "ticker": "RELIANCE",
  "quantity": 10,
  "buy_price": 2850.50,
  "buy_date": "2024-12-01T10:30:00",
  "notes": "Long term investment"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Added 10 shares of RELIANCE to portfolio",
  "portfolio": [
    {
      "ticker": "RELIANCE",
      "quantity": 10,
      "avg_buy_price": 2850.50,
      "buy_date": "2024-12-01T10:30:00",
      "last_updated": "2024-12-04T14:30:00",
      "notes": "Long term investment"
    }
  ]
}
```

**Note:** If stock already exists, it updates average buy price:
- New Avg = (Old Qty × Old Price + New Qty × New Price) / Total Qty

**Example:**
```bash
curl -X POST http://localhost:8000/portfolio/add \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "RELIANCE",
    "quantity": 10,
    "buy_price": 2850.50,
    "notes": "Long term hold"
  }'
```

---

#### `POST /portfolio/sell` - Sell Stock from Portfolio

Sell shares from your portfolio.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| ticker | string | Yes | Stock symbol |
| quantity | integer | Yes | Number of shares to sell (> 0) |
| sell_price | float | Yes | Sale price per share (> 0) |
| sell_date | string | No | Sale date (ISO format) |
| notes | string | No | Optional notes |

**Request Body:**
```json
{
  "ticker": "RELIANCE",
  "quantity": 5,
  "sell_price": 2950.00,
  "notes": "Partial profit booking"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Sold 5 shares of RELIANCE",
  "profit_loss": 497.50,
  "profit_loss_pct": 3.49,
  "portfolio": [
    {
      "ticker": "RELIANCE",
      "quantity": 5,
      "avg_buy_price": 2850.50,
      "buy_date": "2024-12-01T10:30:00",
      "last_updated": "2024-12-04T14:30:00"
    }
  ]
}
```

**Profit/Loss Calculation:**
- Buy Value = Quantity × Avg Buy Price
- Sell Value = Quantity × Sell Price
- P/L = Sell Value - Buy Value
- P/L % = (P/L / Buy Value) × 100

**Example:**
```bash
curl -X POST http://localhost:8000/portfolio/sell \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "RELIANCE",
    "quantity": 5,
    "sell_price": 2950.00
  }'
```

---

#### `GET /portfolio` - Get Portfolio Holdings

View all stocks in your portfolio.

**Parameters:** None

**Response:**
```json
{
  "portfolio": [
    {
      "ticker": "RELIANCE",
      "quantity": 10,
      "avg_buy_price": 2850.50,
      "buy_date": "2024-12-01T10:30:00",
      "last_updated": "2024-12-04T14:30:00",
      "notes": "Long term"
    },
    {
      "ticker": "TCS",
      "quantity": 5,
      "avg_buy_price": 3650.00,
      "buy_date": "2024-11-15T11:00:00",
      "last_updated": "2024-11-15T11:00:00",
      "notes": "IT sector bet"
    }
  ],
  "total_holdings": 2
}
```

**Example:**
```bash
curl http://localhost:8000/portfolio
```

---

#### `GET /portfolio/value` - Portfolio Valuation

Get real-time portfolio value and P&L.

**Parameters:** None

**Response:**
```json
{
  "total_investment": 46755.00,
  "total_current_value": 49502.50,
  "total_profit_loss": 2747.50,
  "total_profit_loss_pct": 5.88,
  "holdings": [
    {
      "ticker": "RELIANCE",
      "quantity": 10,
      "avg_buy_price": 2850.50,
      "current_price": 2920.00,
      "investment": 28505.00,
      "current_value": 29200.00,
      "profit_loss": 695.00,
      "profit_loss_pct": 2.44
    },
    {
      "ticker": "TCS",
      "quantity": 5,
      "avg_buy_price": 3650.00,
      "current_price": 4060.50,
      "investment": 18250.00,
      "current_value": 20302.50,
      "profit_loss": 2052.50,
      "profit_loss_pct": 11.25
    }
  ],
  "number_of_holdings": 2
}
```

**Example:**
```bash
curl http://localhost:8000/portfolio/value
```

---

#### `GET /portfolio/transactions` - Transaction History

Get complete buy/sell transaction history.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| ticker | string | No | Filter by stock symbol |
| transaction_type | string | No | Filter by BUY or SELL |

**Response:**
```json
{
  "transactions": [
    {
      "type": "BUY",
      "ticker": "RELIANCE",
      "quantity": 10,
      "price": 2850.50,
      "date": "2024-12-01T10:30:00",
      "total_value": 28505.00,
      "notes": "Long term"
    },
  ]
}  
```
**Example:**
```bash
# All transactions
curl http://localhost:8000/portfolio/transactions

# Only RELIANCE transactions
curl http://localhost:8000/portfolio/transactions?ticker=RELIANCE

# Only BUY transactions
curl http://localhost:8000/portfolio/transactions?transaction_type=BUY

# RELIANCE sells only
curl "http://localhost:8000/portfolio/transactions?ticker=RELIANCE&transaction_type=SELL"
```

---

### Watchlist

#### `POST /watchlist/add` - Add to Watchlist

Add a stock to your watchlist with optional price alert.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| ticker | string | Yes | Stock symbol |
| target_price | float | No | Alert when price reaches this |
| notes | string | No | Personal notes |

**Request Body:**
```json
{
  "ticker": "INFY",
  "target_price": 1500.00,
  "notes": "Buy on dip below 1450"
}
```

**Response:**
```json
{
  "success": true,
  "message": "INFY added to watchlist",
  "watchlist": [
    {
      "ticker": "INFY",
      "added_date": "2024-12-04T14:30:00",
      "target_price": 1500.00,
      "notes": "Buy on dip below 1450"
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/watchlist/add \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "INFY",
    "target_price": 1500.00,
    "notes": "Good entry point"
  }'
```

---

#### `DELETE /watchlist/{ticker}` - Remove from Watchlist

Remove a stock from watchlist.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| ticker | string | Yes | Stock symbol (path parameter) |

**Response:**
```json
{
  "success": true,
  "message": "INFY removed from watchlist",
  "watchlist": []
}
```

**Example:**
```bash
curl -X DELETE http://localhost:8000/watchlist/INFY
```

---

#### `GET /watchlist` - Get Watchlist

View all stocks in your watchlist.

**Parameters:** None

**Response:**
```json
{
  "watchlist": [
    {
      "ticker": "INFY",
      "added_date": "2024-12-04T14:30:00",
      "target_price": 1500.00,
      "notes": "Buy on dip"
    },
    {
      "ticker": "WIPRO",
      "added_date": "2024-12-03T10:00:00",
      "target_price": 450.00,
      "notes": "Watch for breakout"
    }
  ],
  "total": 2
}
```

**Example:**
```bash
curl http://localhost:8000/watchlist
```

---

### Notifications & Reminders

#### `POST /reminders` - Create Reminder

Set a date-based reminder.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| title | string | Yes | Reminder title |
| message | string | Yes | Reminder message |
| reminder_time | string | Yes | When to trigger (ISO format) |
| reminder_type | string | No | Type (GENERAL, EARNINGS, DIVIDEND) |
| ticker | string | No | Related stock symbol |

**Request Body:**
```json
{
  "title": "RELIANCE Q4 Earnings",
  "message": "Check quarterly results",
  "reminder_time": "2024-12-15T09:30:00",
  "reminder_type": "EARNINGS",
  "ticker": "RELIANCE"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Reminder created",
  "reminder": {
    "id": 1,
    "title": "RELIANCE Q4 Earnings",
    "message": "Check quarterly results",
    "reminder_time": "2024-12-15T09:30:00",
    "type": "EARNINGS",
    "ticker": "RELIANCE",
    "created_at": "2024-12-04T14:30:00",
    "status": "ACTIVE",
    "triggered": false
  }
}
```

**Reminder Types:**
- **GENERAL**: General reminder
- **EARNINGS**: Earnings announcement
- **DIVIDEND**: Dividend payment
- **BUYBACK**: Buyback announcement
- **RIGHTS**: Rights issue

**Example:**
```bash
curl -X POST http://localhost:8000/reminders \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Check TCS Results",
    "message": "Q4 earnings today",
    "reminder_time": "2024-12-20T09:00:00",
    "reminder_type": "EARNINGS",
    "ticker": "TCS"
  }'
```

---

#### `GET /reminders` - Get Reminders

View all reminders.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| status | string | No | Filter by ACTIVE or TRIGGERED |

**Response:**
```json
{
  "reminders": [
    {
      "id": 1,
      "title": "RELIANCE Q4 Earnings",
      "message": "Check results",
      "reminder_time": "2024-12-15T09:30:00",
      "type": "EARNINGS",
      "ticker": "RELIANCE",
      "status": "ACTIVE",
      "triggered": false
    }
  ],
  "total": 1
}
```

**Example:**
```bash
# All reminders
curl http://localhost:8000/reminders

# Only active
curl http://localhost:8000/reminders?status=ACTIVE

# Only triggered
curl http://localhost:8000/reminders?status=TRIGGERED
```

---

#### `POST /email/send` - Send Custom Email

Send a custom email notification.

**Requirements:**
- EMAIL_ADDRESS and EMAIL_PASSWORD must be configured in `.env`
- For Gmail: Use App Password (https://myaccount.google.com/apppasswords)

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| to_email | string | Yes | Recipient email |
| subject | string | Yes | Email subject |
| body | string | Yes | Email body (plain text) |
| body_html | string | No | Email body (HTML) |

**Request Body:**
```json
{
  "to_email": "investor@example.com",
  "subject": "Daily Stock Alert",
  "body": "RELIANCE is up 3% today!",
  "body_html": "<h2>RELIANCE Alert</h2><p>Up 3% today!</p>"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Email sent to investor@example.com"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/email/send \
  -H "Content-Type: application/json" \
  -d '{
    "to_email": "investor@example.com",
    "subject": "Portfolio Update",
    "body": "Your portfolio is up 5% today!"
  }'
```

---

#### `POST /email/portfolio-summary` - Email Portfolio Summary

Send your portfolio summary via email.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| to_email | string | Yes | Recipient email (query param) |

**Response:**
```json
{
  "success": true,
  "message": "Email sent to investor@example.com"
}
```

**Email Contents:**
- Total investment
- Current value
- Overall P&L
- Individual holdings with P&L
- Formatted HTML table

**Example:**
```bash
curl -X POST "http://localhost:8000/email/portfolio-summary?to_email=investor@example.com"
```

---

#### `GET /email/config` - Email Configuration Status

Check if email is configured.

**Parameters:** None

**Response:**
```json
{
  "email_enabled": true,
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "email_address": "your_email@gmail.com",
  "configuration_help": "Set EMAIL_ADDRESS and EMAIL_PASSWORD environment variables",
  "app_password_link": "https://support.google.com/accounts/answer/185833"
}
```

**Example:**
```bash
curl http://localhost:8000/email/config
```

---

## 📋 Data Models

### Stock Information
```json
{
  "ticker": "string",
  "ticker_ns": "string",
  "current_price": "float",
  "previous_close": "float",
  "change": "float",
  "change_percent": "float",
  "volume": "integer",
  "high": "float",
  "low": "float"
}
```

### Prediction Response
```json
{
  "ticker": "string",
  "last_close": "float",
  "predicted_close": "float",
  "predicted_return_pct": "float",
  "signal": "BUY|SELL|HOLD",
  "entry_price": "float",
  "target_price": "float",
  "stop_loss": "float",
  "model_mse": "float",
  "direction_accuracy": "float",
  "feature_importance": "object"
}
```

### Portfolio Holding
```json
{
  "ticker": "string",
  "quantity": "integer",
  "avg_buy_price": "float",
  "buy_date": "string (ISO)",
  "last_updated": "string (ISO)",
  "notes": "string"
}
```

### Transaction
```json
{
  "type": "BUY|SELL",
  "ticker": "string",
  "quantity": "integer",
  "price": "float",
  "date": "string (ISO)",
  "total_value": "float",
  "profit_loss": "float (SELL only)",
  "profit_loss_pct": "float (SELL only)",
  "notes": "string"
}
```

---

## 💡 Examples

### Complete Trading Workflow
```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Check market status
status = requests.get(f"{BASE_URL}/market/status").json()
print(f"Market is: {status['status']}")

# 2. Get market overview
overview = requests.get(f"{BASE_URL}/market/overview").json()
print(f"Sentiment: {overview['market_sentiment']}")

# 3. Find opportunities
gainers = requests.get(f"{BASE_URL}/market/gainers?limit=5").json()
print(f"Top gainer: {gainers[0]['ticker']} (+{gainers[0]['change_percent']}%)")

# 4. Analyze a stock
ticker = "RELIANCE"
analysis = requests.get(f"{BASE_URL}/analysis/{ticker}").json()
print(f"Recommendation: {analysis['technical_signals']['recommendation']}")

# 5. Get prediction
prediction = requests.post(
    f"{BASE_URL}/predict",
    json={"ticker": ticker}
).json()
print(f"Signal: {prediction['signal']}")
print(f"Target: ₹{prediction['target_price']}")

# 6. Multi-timeframe analysis
timeframe = requests.get(f"{BASE_URL}/predict/timeframe/{ticker}").json()
print(f"Weekly trend: {timeframe['weekly']['trend']}")
print(f"Monthly prediction: ₹{timeframe['monthly']['predicted_price']}")

# 7. Add to portfolio
portfolio_add = requests.post(
    f"{BASE_URL}/portfolio/add",
    json={
        "ticker": ticker,
        "quantity": 10,
        "buy_price": prediction['entry_price'],
        "notes": f"ML Signal: {prediction['signal']}"
    }
).json()
print(f"Added to portfolio: {portfolio_add['message']}")

# 8. Add to watchlist
watchlist = requests.post(
    f"{BASE_URL}/watchlist/add",
    json={
        "ticker": "TCS",
        "target_price": 3800.00,
        "notes": "Buy on dip"
    }
).json()

# 9. Get portfolio value
portfolio_value = requests.get(f"{BASE_URL}/portfolio/value").json()
print(f"Portfolio P&L: ₹{portfolio_value['total_profit_loss']} ({portfolio_value['total_profit_loss_pct']}%)")

# 10. Set reminder
reminder = requests.post(
    f"{BASE_URL}/reminders",
    json={
        "title": f"{ticker} Earnings",
        "message": "Check quarterly results",
        "reminder_time": "2024-12-20T09:30:00",
        "reminder_type": "EARNINGS",
        "ticker": ticker
    }
).json()

# 11. Email portfolio summary
email = requests.post(
    f"{BASE_URL}/email/portfolio-summary?to_email=investor@example.com"
).json()
```

---

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG_MODE=True

# CORS Settings
ALLOWED_ORIGINS=*

# Model Configuration
MODEL_N_ESTIMATORS=100
MODEL_MAX_DEPTH=10
MODEL_MIN_SAMPLES_SPLIT=5

# Data Configuration
DEFAULT_PERIOD=2y
MIN_DATA_ROWS=30

# Email Configuration (Optional)
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

### Gmail Setup for Notifications

1. **Enable 2-Factor Authentication**
   - Go to Google Account settings
   - Security → 2-Step Verification

2. **Generate App Password**
   - Visit: https://myaccount.google.com/apppasswords
   - Select "Mail" and your device
   - Copy the 16-character password

3. **Update .env**
```bash
   EMAIL_ADDRESS=your_email@gmail.com
   EMAIL_PASSWORD=xxxx xxxx xxxx xxxx
```

4. **Test Configuration**
```bash
   curl http://localhost:8000/email/config
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Error: Address already in use
# Solution: Use different port
uvicorn app.main:app --reload --port 8001
```

#### 2. Import Errors
```bash
# Error: ModuleNotFoundError
# Solution: Install dependencies
pip install -r requirements.txt
```

#### 3. Stock Data Not Loading
```bash
# Error: Unable to fetch data
# Causes:
- Invalid ticker symbol
- No internet connection
- Yahoo Finance temporarily unavailable

# Solutions:
- Verify ticker is correct (use .NS suffix)
- Check internet connection
- Try different stock
- Wait a few minutes and retry
```

#### 4. Email Not Sending
```bash
# Check configuration
curl http://localhost:8000/email/config

# Common issues:
- App Password not generated
- 2FA not enabled
- Wrong email/password in .env
- Gmail blocking sign-in attempts

# Solution:
1. Enable 2FA
2. Generate App Password
3. Update .env with App Password (not regular password)
4. Restart server
```

#### 5. Portfolio Data Lost
```bash
# Data stored in data/ folder
# Backup before clearing:
cp -r data/ data_backup_$(date +%Y%m%d)/

# Restore from backup:
cp -r data_backup_YYYYMMDD/ data/
```

#### 6. Predictions Taking Too Long
```bash
# First prediction: 30-60 seconds (model training)
# Subsequent predictions: Same time (no caching for ML)

# To speed up:
- Use batch predictions for multiple stocks
- Consider caching predictions with timestamps
```

#### 7. Market Status Wrong
```bash
# Uses IST timezone
# Check system timezone:
date

# Market timings are hardcoded for NSE
# Modify market_schedule.py if needed
```

---

## 📊 Popular Indian Stocks

### Blue Chip Stocks
- **RELIANCE** - Reliance Industries
- **TCS** - Tata Consultancy Services
- **HDFCBANK** - HDFC Bank
- **INFY** - Infosys
- **ICICIBANK** - ICICI Bank

### IT Sector
- TCS, INFY, WIPRO, HCLTECH, TECHM

### Banking
- HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK

### Automotive
- TATAMOTORS, MARUTI, M&M

### FMCG
- HINDUNILVR, ITC, NESTLEIND, BRITANNIA

### Pharma
- SUNPHARMA, DRREDDY, CIPLA, DIVISLAB

### Energy
- RELIANCE, ONGC, BPCL, IOC

---

## 🔐 Security Best Practices

### Production Deployment

1. **API Authentication**
```python
   # Add API key authentication
   from fastapi.security import APIKeyHeader
   
   api_key_header = APIKeyHeader(name="X-API-Key")
   
   @app.get("/protected")
   def protected_route(api_key: str = Depends(api_key_header)):
       if api_key != os.getenv("API_KEY"):
           raise HTTPException(403)
       return {"message": "Authorized"}
```

2. **Rate Limiting**
```python
   from slowapi import Limiter
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   
   @app.get("/predict")
   @limiter.limit("5/minute")
   def predict_with_limit():
       ...
```

3. **HTTPS Only**
```bash
   # Use reverse proxy (nginx) with SSL
   # Or deploy to cloud with managed SSL
```

4. **Environment Variables**
```bash
   # Never commit .env to git
   echo ".env" >> .gitignore
   
   # Use secure secret management in production
   # AWS Secrets Manager, Azure Key Vault, etc.
```

5. **CORS Configuration**
```python
   # Restrict origins in production
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.com"],
       allow_credentials=True,
       allow_methods=["GET", "POST"],
       allow_headers=["*"],
   )
```

---

## 📈 Performance Optimization

### Response Times

| Endpoint | First Call | Cached | Optimization |
|----------|-----------|--------|--------------|
| `/predict` | 30-60s | N/A | Uses 2y optimal period |
| `/market/gainers` | 15-30s | <1s | 5-min cache |
| `/analysis/{ticker}` | 5-10s | <1s | 5-min cache |
| `/portfolio/value` | 2-5s | N/A | Real-time prices |
| `/market/status` | <1s | N/A | Calculation only |

### Caching Strategy
```python
# Market data cached for 5 minutes
cache_duration = timedelta(minutes=5)

# Portfolio data not cached (real-time)
# Predictions not cached (always train new model)
```

### Scaling Tips

1. **Database**: Replace JSON with PostgreSQL/MongoDB
2. **Caching**: Add Redis for frequent queries
3. **Queue**: Use Celery for async ML training
4. **CDN**: Cache static responses
5. **Load Balancer**: Distribute traffic across instances

---

## 🎓 Learning Resources

### Understanding the ML Model

**Features Used (10):**
1. **SMA_5**: 5-day moving average (short-term trend)
2. **SMA_10**: 10-day moving average (medium-term trend)
3. **EMA_10**: Exponential moving average (weighted recent prices)
4. **RSI_14**: Momentum indicator (overbought/oversold)
5. **Volume_Change**: Trading volume percentage change
6. **Return_1d/2d/3d**: Past 1, 2, 3 day returns
7. **Momentum_5**: 5-day price momentum
8. **Volatility_10**: 10-day rolling standard deviation

**Model:** Random Forest Regressor
- 100 decision trees
- Each tree votes on price prediction
- Final prediction = average of all trees
- Handles non-linear relationships well

**Training:**
- Uses 80% data for training, 20% for testing
- Time-series aware split (no shuffle)
- Evaluates on MSE and direction accuracy

### API Design Principles

1. **RESTful**: Standard HTTP methods (GET, POST, DELETE)
2. **Clear Endpoints**: `/resource` or `/resource/{id}`
3. **Consistent Responses**: Always JSON with same structure
4. **Error Handling**: Proper HTTP status codes
5. **Documentation**: Auto-generated with FastAPI

---

## 🎉 Conclusion

This API provides a complete stock trading and portfolio management solution with:

✅ **30+ Endpoints** for comprehensive functionality  
✅ **ML Predictions** with 60-75% direction accuracy  
✅ **Portfolio Tracking** with real-time P&L  
✅ **Market Intelligence** with sentiment analysis  
✅ **Automation** with emails and reminders  
✅ **Multi-Timeframe** analysis for better decisions  

### Next Steps
w
1. **Test All Endpoints**: Use interactive docs at `/docs`
2. **Build Frontend**: Create Flutter/React app
3. **Add Authentication**: Secure your API
4. **Deploy to Cloud**: AWS, GCP, or Azure
5. **Monitor Performance**: Add logging and metrics

---

## 📞 Support

- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Info**: http://localhost:8000/

---

## 📄 License

MIT License - Free to use and modify

---

## ⚠️ Disclaimer

**This is an educational project for learning purposes.**

- Stock predictions are based on historical data and technical indicators
- Past performance does not guarantee future results
- Always do your own research before investing
- Consult financial advisors for investment decisions
- Never invest more than you can afford to lose
- The developers are not responsible for any financial losses

**Use at your own risk!**

---

**Version:** 3.0.0  
**Last Updated:** December 2024  
**Maintained By:** [Your Name/Organization]  
**Repository:** [GitHub URL]</parameter>