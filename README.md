# Indian Stock Market Prediction API - Backend

A production-ready FastAPI backend for predicting Indian stock market prices using Machine Learning.

## 📁 Project Structure

```
stock-prediction-backend/
├── app/
│   ├── __init__.py          # Package initializer
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   └── predictor.py         # ML prediction engine
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Create Project Structure

```bash
# Create project directory
mkdir stock-prediction-backend
cd stock-prediction-backend

# Create app directory
mkdir app

# Create empty __init__.py
touch app/__init__.py
```

### Step 2: Create Files

Create the following files with provided content:
- `app/__init__.py` (empty file)
- `app/main.py`
- `app/models.py`
- `app/predictor.py`
- `requirements.txt`
- `.env`

### Step 3: Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **yfinance**: Stock data fetching
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning

### Step 5: Run the Server

```bash
# From the project root directory
uvicorn app.main:app --reload

# Or specify host and port
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will start at: **http://localhost:8000**

### Step 6: Test the API

Open your browser and visit:
- **API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📚 API Endpoints

### 1. Root Endpoint
```http
GET /
```

Returns API information and available endpoints.

### 2. Health Check
```http
GET /health
```

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running successfully"
}
```

### 3. Stock Prediction
```http
POST /predict
```

Get stock prediction with trading signals.

**Request Body:**
```json
{
  "ticker": "RELIANCE",
  "period": "2y"
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

### 4. Batch Prediction
```http
POST /predict/batch
```

Get predictions for multiple stocks.

**Request:**
```json
{
  "tickers": ["RELIANCE", "TCS", "INFY"],
  "period": "2y"
}
```

## 🧪 Testing with cURL

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "RELIANCE",
    "period": "2y"
  }'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch?period=2y" \
  -H "Content-Type: application/json" \
  -d '["RELIANCE", "TCS", "INFY"]'
```

## 📊 ML Model Details

### Features Generated (10 Technical Indicators)

1. **SMA_5**: 5-day Simple Moving Average
2. **SMA_10**: 10-day Simple Moving Average
3. **EMA_10**: 10-day Exponential Moving Average
4. **RSI_14**: 14-day Relative Strength Index
5. **Volume_Change**: Percentage change in volume
6. **Return_1d**: 1-day lag return
7. **Return_2d**: 2-day lag return
8. **Return_3d**: 3-day lag return
9. **Momentum_5**: 5-day price momentum
10. **Volatility_10**: 10-day rolling standard deviation

### Model Configuration

- **Algorithm**: Random Forest Regressor
- **Number of Trees**: 100
- **Max Depth**: 10
- **Min Samples Split**: 5
- **Train/Test Split**: 80/20 (time-series aware)

### Trading Signals

- **BUY**: Predicted return ≥ +2%
- **HOLD**: Predicted return between -2% and +2%
- **SELL**: Predicted return ≤ -2%

### Risk Management

- **BUY Signal**: Stop-loss at 3% below entry
- **SELL Signal**: Stop-loss at 3% above entry
- **HOLD Signal**: Stop-loss at 2% below entry

## 🎯 Popular Indian Stocks to Try

| Symbol | Company Name |
|--------|-------------|
| RELIANCE | Reliance Industries |
| TCS | Tata Consultancy Services |
| INFY | Infosys |
| HDFCBANK | HDFC Bank |
| ICICIBANK | ICICI Bank |
| TATAMOTORS | Tata Motors |
| WIPRO | Wipro |
| ITC | ITC Limited |
| SBIN | State Bank of India |
| BHARTIARTL | Bharti Airtel |
| KOTAKBANK | Kotak Mahindra Bank |
| LT | Larsen & Toubro |
| ASIANPAINT | Asian Paints |
| MARUTI | Maruti Suzuki |
| SUNPHARMA | Sun Pharmaceutical |

## 🛠️ Troubleshooting

### Issue: ModuleNotFoundError

**Solution:**
```bash
# Ensure you're in the virtual environment
pip install -r requirements.txt
```

### Issue: Port 8000 already in use

**Solution:**
```bash
# Use a different port
uvicorn app.main:app --reload --port 8001
```

### Issue: Cannot fetch stock data

**Possible causes:**
1. Invalid ticker symbol
2. No internet connection
3. Yahoo Finance API temporarily unavailable

**Solution:**
- Verify ticker symbol is correct
- Check internet connection
- Try different stock or wait a few minutes

### Issue: Insufficient data error

**Cause:** Not enough historical data for the selected period

**Solution:**
- Try a longer period (e.g., "2y" instead of "1mo")
- Stock might be newly listed

### Issue: CORS errors from Flutter app

**Solution:**
The backend already has CORS enabled for all origins. If issues persist:
1. Check that the backend is running
2. Verify the URL in Flutter app matches the backend URL
3. For real devices, use your computer's IP address instead of localhost

## 🔒 Security Considerations

### For Production Deployment:

1. **Environment Variables**: Use proper environment variable management
2. **CORS**: Restrict `allow_origins` to specific domains
3. **Rate Limiting**: Implement API rate limiting
4. **Authentication**: Add API key or JWT authentication
5. **HTTPS**: Use SSL/TLS certificates
6. **Input Validation**: Already implemented with Pydantic
7. **Error Handling**: Already implemented with proper error messages

### Example Production CORS Configuration:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📈 Model Performance Metrics

### MSE (Mean Squared Error)
- Lower is better
- Measures average squared difference between predicted and actual values
- Typical values: 50-500 depending on stock price range

### Direction Accuracy
- Percentage of correct directional predictions (up/down)
- Typical values: 55-75%
- Above 60% is considered good for stock prediction

### Feature Importance
- Shows which technical indicators contribute most to predictions
- Helps understand model decision-making
- Typically dominated by moving averages and momentum indicators

## 🚀 Production Deployment

### Option 1: Local Server

```bash
# Install production ASGI server
pip install gunicorn

# Run with Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Option 2: Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t stock-prediction-api .
docker run -p 8000:8000 stock-prediction-api
```

### Option 3: Cloud Platforms

- **Heroku**: Use Procfile
- **AWS**: Deploy to EC2, ECS, or Lambda
- **Google Cloud**: Deploy to Cloud Run or App Engine
- **Azure**: Deploy to App Service

## 📝 API Response Codes

- **200**: Success
- **400**: Bad Request (invalid parameters)
- **404**: Not Found (ticker not found)
- **500**: Internal Server Error

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## ⚠️ Disclaimer

This is an educational project. Stock market predictions are inherently uncertain and should not be the sole basis for investment decisions. Always:
- Do your own research
- Consult with financial advisors
- Understand the risks involved
- Never invest more than you can afford to lose

## 📄 License

MIT License - feel free to use this for educational purposes.

## 🎉 You're Ready!

Your FastAPI backend is now ready to serve stock predictions. Start making predictions and integrate with your Flutter frontend!

For Flutter frontend integration, update the API URL in your Flutter app's `api_service.dart`:
```dart
static const String baseUrl = 'http://YOUR_IP:8000';
```

Happy predicting! 📈