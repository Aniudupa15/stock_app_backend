#!/usr/bin/env python3
"""
Comprehensive Demo Script - All Features
Tests all endpoints including new v3.0 features
"""

import requests
import json
import time
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def test_health():
    """Test basic health"""
    print_header("1. Testing API Health")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_success("API is healthy and running")
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except:
        print_error("Cannot connect to server. Make sure it's running!")
        return False

def test_market_status():
    """Test market status"""
    print_header("2. Testing Market Status")
    try:
        response = requests.get(f"{BASE_URL}/market/status")
        data = response.json()
        print_success(f"Market Status: {data['status']}")
        print_info(f"  Session: {data['session']}")
        print_info(f"  Is Open: {data['is_open']}")
        print_info(f"  Current Time: {data['current_time']}")
        return True
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_predictions():
    """Test stock predictions"""
    print_header("3. Testing Stock Predictions")
    try:
        # Single prediction
        print_info("Testing single prediction...")
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"ticker": "RELIANCE"},
            timeout=90
        )
        data = response.json()
        print_success(f"Prediction: {data['ticker']}")
        print_info(f"  Signal: {data['signal']}")
        print_info(f"  Current: ₹{data['last_close']}")
        print_info(f"  Predicted: ₹{data['predicted_close']}")
        print_info(f"  Return: {data['predicted_return_pct']}%")
        
        # Multi-timeframe
        print_info("\nTesting multi-timeframe analysis...")
        response = requests.get(f"{BASE_URL}/predict/timeframe/TCS")
        data = response.json()
        print_success("Multi-timeframe analysis complete")
        print_info(f"  Weekly: {data['weekly']['trend']}")
        print_info(f"  Monthly: {data['monthly']['trend']}")
        print_info(f"  Yearly: {data['yearly']['trend']}")
        
        return True
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_price_alerts():
    """Test price alert system"""
    print_header("4. Testing Price Alert System")
    try:
        # Create alert
        print_info("Creating price alert...")
        response = requests.post(
            f"{BASE_URL}/alerts/create",
            params={
                "ticker": "RELIANCE",
                "target_price": 3000.00,
                "condition": "above",
                "notes": "Demo alert"
            }
        )
        data = response.json()
        if data['success']:
            print_success(f"Alert created: {data['alert']['ticker']} @ ₹{data['alert']['target_price']}")
            alert_id = data['alert']['id']
            
            # Get alerts
            print_info("\nGetting all alerts...")
            response = requests.get(f"{BASE_URL}/alerts")
            data = response.json()
            print_success(f"Found {data['total']} alerts")
            
            # Check alerts
            print_info("\nChecking alerts...")
            response = requests.post(f"{BASE_URL}/alerts/check")
            data = response.json()
            print_success(f"Checked. Triggered: {data['triggered_alerts']}")
            
            # Delete alert
            print_info(f"\nDeleting alert {alert_id}...")
            response = requests.delete(f"{BASE_URL}/alerts/{alert_id}")
            print_success("Alert deleted")
            
            return True
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_daily_recommendations():
    """Test daily recommendations"""
    print_header("5. Testing Daily Recommendations")
    try:
        print_info("Getting daily recommendations (this may take 1-2 minutes)...")
        response = requests.get(
            f"{BASE_URL}/recommendations/daily?min_score=2",
            timeout=180
        )
        data = response.json()
        
        print_success("Daily recommendations generated")
        print_info(f"  Total analyzed: {data['total_analyzed']}")
        print_info(f"  Strong buys: {data['summary']['strong_buys']}")
        print_info(f"  Buys: {data['summary']['buys']}")
        print_info(f"  Sells: {data['summary']['sells']}")
        
        if data['recommendations']['buy']:
            print_info("\nTop 3 Buy Recommendations:")
            for i, rec in enumerate(data['recommendations']['buy'][:3], 1):
                print(f"  {i}. {rec['ticker']} - {rec['overall_signal']}")
                print(f"     Current: ₹{rec['current_price']}, Score: {rec['score']}")
        
        # Get top picks
        print_info("\nGetting top momentum picks...")
        response = requests.get(f"{BASE_URL}/recommendations/top-picks?category=momentum&limit=3")
        data = response.json()
        print_success(f"Got {data['total']} momentum picks")
        
        return True
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_ipo_features():
    """Test IPO features"""
    print_header("6. Testing IPO Features")
    try:
        # Add IPO
        print_info("Adding sample IPO...")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        two_weeks = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        
        response = requests.post(
            f"{BASE_URL}/ipo/add",
            params={
                "company_name": "Demo Tech Corp",
                "open_date": tomorrow,
                "close_date": next_week,
                "listing_date": two_weeks,
                "price_band": "₹200-250",
                "lot_size": 50,
                "issue_size": "₹300 Cr",
                "category": "Mainboard",
                "sector": "Technology"
            }
        )
        data = response.json()
        if data['success']:
            print_success(f"IPO added: {data['ipo']['company_name']}")
            ipo_id = data['ipo']['id']
            
            # Get upcoming IPOs
            print_info("\nGetting upcoming IPOs...")
            response = requests.get(f"{BASE_URL}/ipo/upcoming?days=30")
            data = response.json()
            print_success(f"Found {data['total']} upcoming IPOs")
            
            # Analyze IPO
            print_info(f"\nAnalyzing IPO {ipo_id}...")
            response = requests.post(f"{BASE_URL}/ipo/{ipo_id}/analyze")
            data = response.json()
            if data['success']:
                print_success(f"Analysis complete")
                print_info(f"  Recommendation: {data['analysis']['recommendation']}")
                print_info(f"  Score: {data['analysis']['score']}")
            
            # Set reminder
            print_info("\nSetting IPO reminder...")
            response = requests.post(
                f"{BASE_URL}/ipo/{ipo_id}/reminder",
                params={
                    "reminder_date": tomorrow,
                    "reminder_type": "OPENING"
                }
            )
            data = response.json()
            if data['success']:
                print_success("Reminder set")
            
            # Get calendar
            print_info("\nGetting IPO calendar...")
            response = requests.get(f"{BASE_URL}/ipo/calendar")
            data = response.json()
            print_success(f"Calendar has {data['total_events']} events")
            
            return True
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_portfolio():
    """Test portfolio features"""
    print_header("7. Testing Portfolio Management")
    try:
        # Add to portfolio
        print_info("Adding stock to portfolio...")
        response = requests.post(
            f"{BASE_URL}/portfolio/add",
            json={
                "ticker": "TCS",
                "quantity": 10,
                "buy_price": 3650.00,
                "notes": "Demo purchase"
            }
        )
        data = response.json()
        if data['success']:
            print_success(f"Added to portfolio: {data['message']}")
            
            # Get portfolio value
            print_info("\nGetting portfolio value...")
            response = requests.get(f"{BASE_URL}/portfolio/value")
            data = response.json()
            print_success("Portfolio valuation complete")
            print_info(f"  Investment: ₹{data['total_investment']}")
            print_info(f"  Current Value: ₹{data['total_current_value']}")
            print_info(f"  P&L: ₹{data['total_profit_loss']} ({data['total_profit_loss_pct']}%)")
            
            return True
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_email_notifications():
    """Test email notifications"""
    print_header("8. Testing Email Notifications")
    try:
        # Check email config
        print_info("Checking email configuration...")
        response = requests.get(f"{BASE_URL}/email/config")
        data = response.json()
        
        if data['email_enabled']:
            print_success("Email is configured")
            print_info(f"  SMTP: {data['smtp_server']}")
            print_info(f"  Email: {data['email_address']}")
            
            # Note: We won't actually send email in demo to avoid spam
            print_info("\nEmail features available:")
            print_info("  - POST /email/portfolio-summary")
            print_info("  - Automatic alerts when price targets hit")
            print_info("  - IPO reminders")
        else:
            print_error("Email not configured")
            print_info("Set EMAIL_ADDRESS and EMAIL_PASSWORD in .env to enable")
        
        return True
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_analysis():
    """Test stock analysis"""
    print_header("9. Testing Stock Analysis")
    try:
        print_info("Getting comprehensive analysis...")
        response = requests.get(f"{BASE_URL}/analysis/INFY?period=6mo")
        data = response.json()
        
        print_success(f"Analysis complete for {data['ticker']}")
        print_info(f"  Current Price: ₹{data['current_price']}")
        print_info(f"  RSI: {data['indicators']['rsi']} ({data['indicators']['rsi_signal']})")
        print_info(f"  MACD: {data['indicators']['macd_trend']}")
        print_info(f"  Recommendation: {data['technical_signals']['recommendation']}")
        
        # Compare stocks
        print_info("\nComparing multiple stocks...")
        response = requests.get(
            f"{BASE_URL}/compare",
            params=[
                ("tickers", "RELIANCE"),
                ("tickers", "TCS"),
                ("tickers", "INFY")
            ]
        )
        data = response.json()
        print_success(f"Compared {len(data)} stocks")
        
        return True
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def main():
    """Run complete demo"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║          Stock Market API v3.0 - Comprehensive Feature Demo                 ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    tests = [
        ("API Health", test_health),
        ("Market Status", test_market_status),
        ("Predictions & Multi-Timeframe", test_predictions),
        ("Price Alert System", test_price_alerts),
        ("Daily Recommendations", test_daily_recommendations),
        ("IPO Features", test_ipo_features),
        ("Portfolio Management", test_portfolio),
        ("Email Notifications", test_email_notifications),
        ("Stock Analysis", test_analysis),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            if result:
                time.sleep(1)  # Small delay between tests
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Demo interrupted by user{Colors.END}")
            break
        except Exception as e:
            print_error(f"Test failed: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Demo Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        if result:
            print_success(f"{name}")
        else:
            print_error(f"{name}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All features working! API is fully functional.{Colors.END}")
        print(f"\n{Colors.BOLD}Available Features:{Colors.END}")
        print("  ✓ Stock Predictions (ML-based)")
        print("  ✓ Multi-Timeframe Analysis (Weekly/Monthly/Yearly)")
        print("  ✓ Price Alerts with Email Notifications")
        print("  ✓ Daily Buy/Sell Recommendations")
        print("  ✓ IPO Tracking & Analysis")
        print("  ✓ Portfolio Management")
        print("  ✓ Market Status & Holidays")
        print("  ✓ Technical Analysis")
        print("  ✓ Email Notifications")
        print(f"\n{Colors.BOLD}Access:{Colors.END}")
        print(f"  • Interactive Docs: {BASE_URL}/docs")
        print(f"  • API Status: {BASE_URL}/health")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Some features need attention{Colors.END}")
        print("\nPlease check:")
        print("  • Server is running (uvicorn app.main:app --reload)")
        print("  • All dependencies installed (pip install -r requirements.txt)")
        print("  • All files are present (verify_setup.py)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Demo terminated by user{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.END}")