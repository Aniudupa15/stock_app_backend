#!/usr/bin/env python3
"""
Verify alert endpoints are properly configured
"""

import sys
from pathlib import Path

def check_main_py():
    """Check main.py for alert endpoints"""
    main_file = Path("app/main.py")
    
    if not main_file.exists():
        print("✗ app/main.py not found!")
        return False
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    checks = {
        "Import PriceAlertSystem": "from price_alert_system import PriceAlertSystem" in content,
        "/alerts/create endpoint": '@app.post(\n    "/alerts/create"' in content or '@app.post("/alerts/create")' in content,
        "/alerts GET endpoint": '@app.get(\n    "/alerts"' in content or '@app.get("/alerts")' in content,
        "/alerts/check endpoint": '@app.post(\n    "/alerts/check"' in content or '@app.post("/alerts/check")' in content,
        "PriceAlertSystem() usage": "PriceAlertSystem()" in content,
    }
    
    print("Checking app/main.py...")
    print("-" * 50)
    
    all_ok = True
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")
        if not result:
            all_ok = False
    
    return all_ok

def check_price_alert_file():
    """Check if price_alert_system.py exists"""
    alert_file = Path("app/price_alert_system.py")
    
    print("\nChecking app/price_alert_system.py...")
    print("-" * 50)
    
    if not alert_file.exists():
        print("✗ File does not exist!")
        return False
    
    with open(alert_file, 'r') as f:
        content = f.read()
    
    checks = {
        "PriceAlertSystem class": "class PriceAlertSystem" in content,
        "create_alert method": "def create_alert" in content,
        "get_alerts method": "def get_alerts" in content,
        "check_alerts method": "def check_alerts" in content,
        "delete_alert method": "def delete_alert" in content,
    }
    
    all_ok = True
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")
        if not result:
            all_ok = False
    
    return all_ok

def test_import():
    """Try importing the module"""
    print("\nTesting import...")
    print("-" * 50)
    
    try:
        sys.path.insert(0, 'app')
        from price_alert_system import PriceAlertSystem
        print("✓ Successfully imported PriceAlertSystem")
        
        # Try creating instance
        alert_system = PriceAlertSystem()
        print("✓ Successfully created PriceAlertSystem instance")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def provide_solution():
    """Provide solution steps"""
    print("\n" + "="*70)
    print("SOLUTION")
    print("="*70)
    
    print("\nStep 1: Ensure app/price_alert_system.py exists")
    print("  Copy the PriceAlertSystem class code to this file")
    
    print("\nStep 2: Update app/main.py")
    print("  Add import after other imports:")
    print("  from price_alert_system import PriceAlertSystem")
    
    print("\nStep 3: Add endpoints to app/main.py")
    print("  Add these 5 endpoints after the notifications section:")
    
    print("""
# ==================== PRICE ALERTS ====================

@app.post("/alerts/create", tags=["Price Alerts"])
async def create_price_alert(
    ticker: str,
    target_price: float,
    condition: str = Query(..., regex="^(above|below)$"),
    email: Optional[str] = None,
    notes: Optional[str] = None
):
    try:
        alert_system = PriceAlertSystem()
        result = alert_system.create_alert(ticker, target_price, condition, email, notes)
        return result
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(500, detail=str(e))

@app.get("/alerts", tags=["Price Alerts"])
async def get_price_alerts(
    status: Optional[str] = None,
    ticker: Optional[str] = None
):
    try:
        alert_system = PriceAlertSystem()
        alerts = alert_system.get_alerts(status, ticker)
        return {"alerts": alerts, "total": len(alerts)}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/alerts/check", tags=["Price Alerts"])
async def check_price_alerts():
    try:
        alert_system = PriceAlertSystem()
        notif_service = NotificationService()
        triggered = alert_system.check_alerts()
        
        for alert in triggered:
            if alert.get('email'):
                notif_service.send_price_alert_email(
                    alert['email'],
                    alert['ticker'],
                    alert['triggered_price'],
                    alert['target_price'],
                    alert['condition']
                )
        
        return {
            "checked_at": datetime.now().isoformat(),
            "triggered_alerts": len(triggered),
            "alerts": triggered
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.delete("/alerts/{alert_id}", tags=["Price Alerts"])
async def delete_price_alert(alert_id: int):
    try:
        alert_system = PriceAlertSystem()
        result = alert_system.delete_alert(alert_id)
        return result
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/alerts/summary", tags=["Price Alerts"])
async def get_alerts_summary():
    try:
        alert_system = PriceAlertSystem()
        summary = alert_system.get_alert_summary()
        return summary
    except Exception as e:
        raise HTTPException(500, detail=str(e))
""")
    
    print("\nStep 4: Restart server")
    print("  uvicorn app.main:app --reload")
    
    print("\nStep 5: Test")
    print("  curl -X POST 'http://localhost:8000/alerts/create?ticker=RELIANCE&target_price=3000&condition=above'")

def main():
    print("\n🔍 Alert Endpoint Verification\n")
    
    file_ok = check_price_alert_file()
    main_ok = check_main_py()
    import_ok = test_import()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if file_ok and main_ok and import_ok:
        print("\n✓ Everything looks good!")
        print("\nIf endpoints still don't work:")
        print("  1. Check server is running")
        print("  2. Restart server: uvicorn app.main:app --reload")
        print("  3. Check http://localhost:8000/docs for /alerts endpoints")
    else:
        print("\n✗ Issues found:")
        if not file_ok:
            print("  • price_alert_system.py has issues")
        if not main_ok:
            print("  • main.py missing alert endpoints")
        if not import_ok:
            print("  • Cannot import PriceAlertSystem")
        
        provide_solution()
    
    return 0 if (file_ok and main_ok and import_ok) else 1

if __name__ == "__main__":
    sys.exit(main())