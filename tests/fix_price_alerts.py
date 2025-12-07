#!/usr/bin/env python3
"""
Fix and test Price Alert System
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_price_alerts():
    """Test price alert system step by step"""
    print("="*70)
    print("Debugging Price Alert System")
    print("="*70)
    
    # Test 1: Check if endpoint exists
    print("\n1. Testing endpoint availability...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            endpoints = data.get('endpoints', {})
            
            # Check for alert endpoints
            alert_endpoints = [k for k in endpoints.keys() if 'alert' in k.lower()]
            
            if alert_endpoints:
                print(f"✓ Found {len(alert_endpoints)} alert endpoints")
                for endpoint in alert_endpoints:
                    print(f"  • {endpoint}")
            else:
                print("✗ No alert endpoints found in API")
                print("\nAvailable endpoints:")
                for endpoint in list(endpoints.keys())[:10]:
                    print(f"  • {endpoint}")
        else:
            print(f"✗ API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server")
        print("Make sure server is running: uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test 2: Try creating an alert
    print("\n2. Testing alert creation...")
    try:
        response = requests.post(
            f"{BASE_URL}/alerts/create",
            params={
                "ticker": "RELIANCE",
                "target_price": 3000.00,
                "condition": "above",
                "notes": "Test alert"
            },
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Alert created successfully!")
            print(json.dumps(data, indent=2))
            return True
        elif response.status_code == 404:
            print("✗ Endpoint not found (404)")
            print("The /alerts/create endpoint doesn't exist")
            return False
        elif response.status_code == 500:
            print("✗ Server error (500)")
            print("Response:", response.text)
            return False
        else:
            print(f"✗ Unexpected status: {response.status_code}")
            print("Response:", response.text)
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def check_files():
    """Check if required files exist"""
    print("\n3. Checking required files...")
    
    import os
    from pathlib import Path
    
    files_to_check = [
        "app/price_alert_system.py",
        "app/main.py",
        "data"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} MISSING")
            all_exist = False
    
    # Check if price_alert_system is imported in main.py
    if Path("app/main.py").exists():
        with open("app/main.py", 'r') as f:
            content = f.read()
            if "from price_alert_system import PriceAlertSystem" in content:
                print("✓ PriceAlertSystem imported in main.py")
            else:
                print("✗ PriceAlertSystem NOT imported in main.py")
                all_exist = False
            
            if "/alerts/create" in content:
                print("✓ /alerts/create endpoint defined in main.py")
            else:
                print("✗ /alerts/create endpoint NOT found in main.py")
                all_exist = False
    
    return all_exist

def provide_fix():
    """Provide fix instructions"""
    print("\n" + "="*70)
    print("HOW TO FIX")
    print("="*70)
    
    print("\n1. Make sure app/price_alert_system.py exists")
    print("   • Create the file with PriceAlertSystem class")
    
    print("\n2. Update app/main.py imports:")
    print("   Add this line near the top:")
    print("   from price_alert_system import PriceAlertSystem")
    
    print("\n3. Make sure alert endpoints are in app/main.py")
    print("   Should have these endpoints:")
    print("   • POST /alerts/create")
    print("   • GET /alerts")
    print("   • POST /alerts/check")
    print("   • DELETE /alerts/{id}")
    print("   • GET /alerts/summary")
    
    print("\n4. Restart the server:")
    print("   uvicorn app.main:app --reload")
    
    print("\n5. Test again:")
    print("   python fix_price_alerts.py")

if __name__ == "__main__":
    print("\n🔧 Price Alert System Diagnostics\n")
    
    # Run tests
    api_ok = test_price_alerts()
    files_ok = check_files()
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    if api_ok and files_ok:
        print("\n✓ Price Alert System is working!")
        print("\nYou can now:")
        print("  • Create alerts: POST /alerts/create")
        print("  • View alerts: GET /alerts")
        print("  • Check alerts: POST /alerts/check")
    else:
        print("\n✗ Price Alert System needs fixes")
        provide_fix()
        
        print("\n" + "="*70)
        print("QUICK FIX SCRIPT")
        print("="*70)
        print("""
Create a file 'quick_fix.py' with this code:

```python
# Read main.py
with open('app/main.py', 'r') as f:
    content = f.read()

# Check if import exists
if 'from price_alert_system import PriceAlertSystem' not in content:
    # Find the imports section
    import_line = 'from notification_service import NotificationService'
    if import_line in content:
        content = content.replace(
            import_line,
            import_line + '\\nfrom price_alert_system import PriceAlertSystem'
        )
        
        # Write back
        with open('app/main.py', 'w') as f:
            f.write(content)
        
        print('✓ Added import to main.py')
        print('Now restart server: uvicorn app.main:app --reload')
    else:
        print('Could not find import location')
else:
    print('Import already exists')
```

Then run: python quick_fix.py
""")