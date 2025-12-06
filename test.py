#!/usr/bin/env python3
"""
Complete API Testing Script for Enhanced Stock Market API
Run this to test all endpoints and features
"""

import requests
import json
from typing import Dict, List
import time

BASE_URL = "http://localhost:8000"

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def print_json(data: Dict or List, indent: int = 2):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=indent))

def test_health():
    """Test health check endpoint"""
    print_header("Testing Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_success("API is healthy!")
            print_json(response.json())
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Cannot connect to API: {e}")
        print_info("Make sure the server is running: uvicorn app.main:app --reload")
        return False

def test_prediction():
    """Test stock prediction endpoint"""
    print_header("Testing Stock Prediction (ML)")
    try:
        payload = {
            "ticker": "RELIANCE",
            "period": "2y"
        }
        print_info(f"Predicting: {payload['ticker']} (Period: {payload['period']})")
        print_info("This may take 30-60 seconds...")
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=120)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Prediction completed in {elapsed:.2f}s")
            print(f"\n{Colors.BOLD}Results:{Colors.END}")
            print(f"  Ticker: {data['ticker']}")
            print(f"  Last Close: ₹{data['last_close']}")
            print(f"  Predicted: ₹{data['predicted_close']}")
            print(f"  Expected Return: {data['predicted_return_pct']}%")
            
            signal_color = Colors.GREEN if data['signal'] == 'BUY' else Colors.RED if data['signal'] == 'SELL' else Colors.YELLOW
            print(f"  Signal: {signal_color}{data['signal']}{Colors.END}")
            print(f"  Entry: ₹{data['entry_price']}")
            print(f"  Target: ₹{data['target_price']}")
            print(f"  Stop Loss: ₹{data['stop_loss']}")
            print(f"  Model Accuracy: {data['direction_accuracy']}%")
            return True
        else:
            print_error(f"Prediction failed: {response.text}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_gainers():
    """Test top gainers endpoint"""
    print_header("Testing Top Gainers")
    try:
        print_info("Fetching top 5 gainers...")
        response = requests.get(f"{BASE_URL}/market/gainers?limit=5", timeout=60)
        
        if response.status_code == 200:
            gainers = response.json()
            print_success(f"Found {len(gainers)} top gainers")
            print(f"\n{Colors.BOLD}{'Rank':<6}{'Ticker':<12}{'Price':<12}{'Change':<12}{'Volume'}{Colors.END}")
            print("-" * 70)
            for i, stock in enumerate(gainers, 1):
                print(f"{i:<6}{stock['ticker']:<12}₹{stock['current_price']:<10}{Colors.GREEN}+{stock['change_percent']}%{Colors.END:<12}{stock['volume']:,}")
            return True
        else:
            print_error(f"Failed to fetch gainers: {response.text}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_losers():
    """Test top losers endpoint"""
    print_header("Testing Top Losers")
    try:
        print_info("Fetching top 5 losers...")
        response = requests.get(f"{BASE_URL}/market/losers?limit=5", timeout=60)
        
        if response.status_code == 200:
            losers = response.json()
            print_success(f"Found {len(losers)} top losers")
            print(f"\n{Colors.BOLD}{'Rank':<6}{'Ticker':<12}{'Price':<12}{'Change':<12}{'Volume'}{Colors.END}")
            print("-" * 70)
            for i, stock in enumerate(losers, 1):
                print(f"{i:<6}{stock['ticker']:<12}₹{stock['current_price']:<10}{Colors.RED}{stock['change_percent']}%{Colors.END:<12}{stock['volume']:,}")
            return True
        else:
            print_error(f"Failed to fetch losers: {response.text}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_market_overview():
    """Test market overview endpoint"""
    print_header("Testing Market Overview")
    try:
        print_info("Fetching market overview...")
        response = requests.get(f"{BASE_URL}/market/overview", timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Market overview retrieved")
            
            sentiment_color = Colors.GREEN if data['market_sentiment'] == 'Bullish' else Colors.RED if data['market_sentiment'] == 'Bearish' else Colors.YELLOW
            
            print(f"\n{Colors.BOLD}Market Statistics:{Colors.END}")
            print(f"  Advancing: {Colors.GREEN}{data['advancing']}{Colors.END}")
            print(f"  Declining: {Colors.RED}{data['declining']}{Colors.END}")
            print(f"  Unchanged: {data['unchanged']}")
            print(f"  A/D Ratio: {data['advance_decline_ratio']}")
            print(f"  Sentiment: {sentiment_color}{data['market_sentiment']}{Colors.END}")
            print(f"  Total Volume: {data['total_volume']:,}")
            print(f"  Stocks Analyzed: {data['stocks_analyzed']}")
            return True
        else:
            print_error(f"Failed to fetch overview: {response.text}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_market_status():
    """Test market status endpoint (currently disabled)"""
    print_header("Testing Market Status (Disabled)")
    try:
        response = requests.get(f"{BASE_URL}/market/status", timeout=10)
        if response.status_code == 501:
            print_success("Market Status is correctly disabled (HTTP 501)")
            return True
        else:
            print_error(f"Market Status returned unexpected status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_holidays():
    """Test market holidays endpoint (currently disabled)"""
    print_header("Testing Market Holidays (Disabled)")
    try:
        response = requests.get(f"{BASE_URL}/market/holidays", timeout=10)
        if response.status_code == 501:
            print_success("Market Holidays is correctly disabled (HTTP 501)")
            return True
        else:
            print_error(f"Market Holidays returned unexpected status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False
        
def test_stock_analysis():
    """Test comprehensive stock analysis endpoint"""
    print_header("Testing Stock Analysis")
    try:
        ticker = "TCS"
        print_info(f"Analyzing {ticker} (1 year data)...")
        response = requests.get(f"{BASE_URL}/analysis/{ticker}?period=1y", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Analysis completed for {data['ticker']}")
            
            print(f"\n{Colors.BOLD}Price Information:{Colors.END}")
            print(f"  Current: ₹{data['current_price']}")
            print(f"  52W High: ₹{data['price_stats']['52_week_high']}")
            print(f"  52W Low: ₹{data['price_stats']['52_week_low']}")
            
            print(f"\n{Colors.BOLD}Technical Indicators:{Colors.END}")
            print(f"  RSI: {data['indicators']['rsi']} ({data['indicators']['rsi_signal']})")
            print(f"  MACD Trend: {data['indicators']['macd_trend']}")
            
            print(f"\n{Colors.BOLD}Moving Averages:{Colors.END}")
            print(f"  SMA 20: ₹{data['moving_averages']['sma_20']}")
            print(f"  SMA 50: ₹{data['moving_averages']['sma_50']}")
            print(f"  SMA 200: ₹{data['moving_averages']['sma_200']}")
            
            print(f"\n{Colors.BOLD}Trend Analysis:{Colors.END}")
            print(f"  Short-term: {data['trend']['short_term']}")
            print(f"  Medium-term: {data['trend']['medium_term']}")
            print(f"  Strength: {data['trend']['strength']}%")
            
            recommendation = data['technical_signals']['recommendation']
            rec_color = Colors.GREEN if 'Buy' in recommendation else Colors.RED if 'Sell' in recommendation else Colors.YELLOW
            print(f"\n{Colors.BOLD}Recommendation:{Colors.END} {rec_color}{recommendation}{Colors.END}")
            
            return True
        else:
            print_error(f"Analysis failed: {response.text}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_comparison():
    """Test stock comparison endpoint"""
    print_header("Testing Stock Comparison")
    try:
        tickers = ["RELIANCE", "TCS", "INFY"]
        period = "6mo"
        print_info(f"Comparing: {', '.join(tickers)} (Period: {period})")
        
        # FIX: The API endpoint uses Query parameters. Construct the URL accordingly.
        query_params = "&".join([f"tickers={t}" for t in tickers])
        
        url = f"{BASE_URL}/compare?{query_params}&period={period}"
        
        # Send a POST request with the correctly formed query string
        response = requests.post(url, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Comparison completed for {len(data)} stocks")
            
            print(f"\n{Colors.BOLD}{'Ticker':<15}{'Price':<12}{'Return':<12}{'Volatility':<12}{'Avg Volume'}{Colors.END}")
            print("-" * 70)
            for stock in data:
                return_color = Colors.GREEN if stock['period_return'] > 0 else Colors.RED
                print(f"{stock['ticker']:<15}₹{stock['current_price']:<10}{return_color}{stock['period_return']:>6.2f}%{Colors.END:<12}{stock['volatility']:>6.2f}%    {stock['avg_volume']:>12,}")
            return True
        else:
            print_error(f"Comparison failed: {response.text}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print_header("Testing Batch Prediction")
    try:
        tickers = ["RELIANCE", "TCS"]
        print_info(f"Batch predicting: {', '.join(tickers)}")
        print_info("This may take 1-2 minutes...")
        
        response = requests.post(f"{BASE_URL}/predict/batch?period=2y", json=tickers, timeout=180)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Batch prediction completed for {len(data)} stocks")
            
            print(f"\n{Colors.BOLD}{'Ticker':<15}{'Current':<12}{'Predicted':<12}{'Return':<12}{'Signal'}{Colors.END}")
            print("-" * 70)
            for pred in data:
                signal_color = Colors.GREEN if pred['signal'] == 'BUY' else Colors.RED if pred['signal'] == 'SELL' else Colors.YELLOW
                return_color = Colors.GREEN if pred['predicted_return_pct'] > 0 else Colors.RED
                print(f"{pred['ticker']:<15}₹{pred['last_close']:<10}₹{pred['predicted_close']:<10}{return_color}{pred['predicted_return_pct']:>6.2f}%{Colors.END:<12}{signal_color}{pred['signal']}{Colors.END}")
            return True
        else:
            print_error(f"Batch prediction failed: {response.text}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║       Enhanced Indian Stock Market API - Complete Test Suite    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    # Test health first
    if not test_health():
        print_error("\nAPI is not running. Please start the server first.")
        print_info("Run: uvicorn app.main:app --reload")
        return
    
    # Track results
    results = []
    
    # Run all tests
    tests = [
        ("Market Overview", test_market_overview),
        ("Top Gainers", test_gainers),
        ("Top Losers", test_losers),
        ("Stock Analysis", test_stock_analysis),
        ("Stock Comparison", test_comparison),
        ("ML Prediction", test_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Market Status", test_market_status),
        ("Market Holidays", test_holidays),
    ]
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            time.sleep(1) # Small delay between tests
        except Exception as e:
            print_error(f"Test failed with exception: {e}")
            results.append((name, False))
    
    # Print summary
    print_header("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        if result:
            print_success(f"{name}")
        else:
            print_error(f"{name}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed! API is fully functional.{Colors.END}")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️ Some tests failed. Check the logs above.{Colors.END}")

if __name__ == "__main__":
    try:
        main()                  
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Tests interrupted by user.{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.END}")