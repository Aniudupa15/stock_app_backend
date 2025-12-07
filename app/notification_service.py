# import smtplib
# import json
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from typing import List, Dict, Optional
# from datetime import datetime
# from pathlib import Path
# import os

# class NotificationService:
#     """
#     Notification service for sending alerts via email and storing reminders
#     """
    
    
#     def _init_files(self):
#         """Initialize JSON files"""
#         if not self.reminders_file.exists():
#             self._save_json(self.reminders_file, [])
        
#         if not self.notifications_file.exists():
#             self._save_json(self.notifications_file, [])
    
#     def _load_json(self, filepath: Path) -> any:
#         """Load JSON file"""
#         try:
#             with open(filepath, 'r') as f:
#                 return json.load(f)
#         except:
#             return []
    
#     def _save_json(self, filepath: Path, data: any):
#         """Save to JSON file"""
#         try:
#             with open(filepath, 'w') as f:
#                 json.dump(data, f, indent=2)
#         except Exception as e:
#             print(f"Error saving {filepath}: {e}")
    
#     # ==================== EMAIL ====================
    
#     def send_email(self, to_email: str, subject: str, body: str, 
#                    body_html: Optional[str] = None) -> Dict:
#         """
#         Send email notification
        
#         Note: Requires EMAIL_ADDRESS and EMAIL_PASSWORD environment variables
#         For Gmail, use App Password: https://support.google.com/accounts/answer/185833
#         """
#         if not self.email_enabled:
#             return {
#                 "success": False,
#                 "message": "Email not configured. Set EMAIL_ADDRESS and EMAIL_PASSWORD environment variables."
#             }
        
#         try:
#             # Create message
#             msg = MIMEMultipart('alternative')
#             msg['From'] = self.email_address
#             msg['To'] = to_email
#             msg['Subject'] = subject
            
#             # Add text version
#             part1 = MIMEText(body, 'plain')
#             msg.attach(part1)
            
#             # Add HTML version if provided
#             if body_html:
#                 part2 = MIMEText(body_html, 'html')
#                 msg.attach(part2)
            
#             # Send email
#             with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
#                 server.starttls()
#                 server.login(self.email_address, self.email_password)
#                 server.send_message(msg)
            
#             # Log notification
#             self._log_notification("EMAIL", to_email, subject, body, True)
            
#             return {
#                 "success": True,
#                 "message": f"Email sent to {to_email}"
#             }
        
#         except Exception as e:
#             self._log_notification("EMAIL", to_email, subject, body, False, str(e))
#             return {
#                 "success": False,
#                 "message": f"Failed to send email: {str(e)}"
#             }
    
#     def send_price_alert_email(self, to_email: str, ticker: str, 
#                               current_price: float, target_price: float,
#                               condition: str) -> Dict:
#         """Send price alert email"""
#         subject = f"🚨 Price Alert: {ticker}"
        
#         body = f"""
# Price Alert for {ticker}

# Current Price: ₹{current_price}
# Target Price: ₹{target_price}
# Condition: {condition}

# The stock has {'reached' if condition == 'reached' else 'crossed'} your target price.

# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# ---
# Indian Stock Market Prediction System
#         """
        
#         body_html = f"""
#         <html>
#             <body style="font-family: Arial, sans-serif;">
#                 <h2 style="color: #d32f2f;">🚨 Price Alert: {ticker}</h2>
#                 <div style="background: #f5f5f5; padding: 15px; border-radius: 5px;">
#                     <p><strong>Current Price:</strong> ₹{current_price}</p>
#                     <p><strong>Target Price:</strong> ₹{target_price}</p>
#                     <p><strong>Condition:</strong> {condition}</p>
#                 </div>
#                 <p>The stock has <strong>{('reached' if condition == 'reached' else 'crossed')}</strong> your target price.</p>
#                 <p><small>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
#                 <hr>
#                 <p><small>Indian Stock Market Prediction System</small></p>
#             </body>
#         </html>
#         """
        
#         return self.send_email(to_email, subject, body, body_html)
    
#     def send_portfolio_summary_email(self, to_email: str, portfolio_data: Dict) -> Dict:
#         """Send portfolio summary email"""
#         subject = "📊 Your Portfolio Summary"
        
#         body = f"""
# Portfolio Summary

# Total Investment: ₹{portfolio_data['total_investment']}
# Current Value: ₹{portfolio_data['total_current_value']}
# Profit/Loss: ₹{portfolio_data['total_profit_loss']} ({portfolio_data['total_profit_loss_pct']}%)

# Number of Holdings: {portfolio_data['number_of_holdings']}

# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# ---
# Indian Stock Market Prediction System
#         """
        
#         holdings_html = ""
#         for holding in portfolio_data['holdings']:
#             color = "#4caf50" if holding['profit_loss'] >= 0 else "#f44336"
#             holdings_html += f"""
#             <tr>
#                 <td>{holding['ticker']}</td>
#                 <td>{holding['quantity']}</td>
#                 <td>₹{holding['avg_buy_price']}</td>
#                 <td>₹{holding['current_price']}</td>
#                 <td style="color: {color};">₹{holding['profit_loss']} ({holding['profit_loss_pct']}%)</td>
#             </tr>
#             """
        
#         body_html = f"""
#         <html>
#             <body style="font-family: Arial, sans-serif;">
#                 <h2>📊 Your Portfolio Summary</h2>
#                 <div style="background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
#                     <p><strong>Total Investment:</strong> ₹{portfolio_data['total_investment']}</p>
#                     <p><strong>Current Value:</strong> ₹{portfolio_data['total_current_value']}</p>
#                     <p><strong>Profit/Loss:</strong> <span style="color: {'#4caf50' if portfolio_data['total_profit_loss'] >= 0 else '#f44336'};">
#                         ₹{portfolio_data['total_profit_loss']} ({portfolio_data['total_profit_loss_pct']}%)
#                     </span></p>
#                 </div>
                
#                 <h3>Holdings ({portfolio_data['number_of_holdings']})</h3>
#                 <table style="width: 100%; border-collapse: collapse;">
#                     <thead>
#                         <tr style="background: #f5f5f5;">
#                             <th style="padding: 10px; text-align: left;">Stock</th>
#                             <th style="padding: 10px; text-align: left;">Qty</th>
#                             <th style="padding: 10px; text-align: left;">Avg Buy</th>
#                             <th style="padding: 10px; text-align: left;">Current</th>
#                             <th style="padding: 10px; text-align: left;">P&L</th>
#                         </tr>
#                     </thead>
#                     <tbody>
#                         {holdings_html}
#                     </tbody>
#                 </table>
                
#                 <p><small>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
#                 <hr>
#                 <p><small>Indian Stock Market Prediction System</small></p>
#             </body>
#         </html>
#         """
        
#         return self.send_email(to_email, subject, body, body_html)
    
#     # ==================== REMINDERS ====================
    
#     def create_reminder(self, title: str, message: str, 
#                        reminder_time: str, reminder_type: str = "GENERAL",
#                        ticker: Optional[str] = None) -> Dict:
#         """Create a new reminder"""
#         reminders = self._load_json(self.reminders_file)
        
#         reminder = {
#             "id": len(reminders) + 1,
#             "title": title,
#             "message": message,
#             "reminder_time": reminder_time,
#             "type": reminder_type,
#             "ticker": ticker,
#             "created_at": datetime.now().isoformat(),
#             "status": "ACTIVE",
#             "triggered": False
#         }
        
#         reminders.append(reminder)
#         self._save_json(self.reminders_file, reminders)
        
#         return {
#             "success": True,
#             "message": "Reminder created",
#             "reminder": reminder
#         }
    
#     def get_reminders(self, status: Optional[str] = None) -> List[Dict]:
#         """Get all reminders, optionally filtered by status"""
#         reminders = self._load_json(self.reminders_file)
        
#         if status:
#             reminders = [r for r in reminders if r['status'] == status.upper()]
        
#         return reminders
    
#     def get_pending_reminders(self) -> List[Dict]:
#         """Get reminders that need to be triggered"""
#         reminders = self._load_json(self.reminders_file)
#         now = datetime.now()
        
#         pending = []
#         for reminder in reminders:
#             if reminder['status'] == 'ACTIVE' and not reminder['triggered']:
#                 reminder_time = datetime.fromisoformat(reminder['reminder_time'])
#                 if now >= reminder_time:
#                     pending.append(reminder)
        
#         return pending
    
#     def mark_reminder_triggered(self, reminder_id: int) -> Dict:
#         """Mark a reminder as triggered"""
#         reminders = self._load_json(self.reminders_file)
        
#         for reminder in reminders:
#             if reminder['id'] == reminder_id:
#                 reminder['triggered'] = True
#                 reminder['triggered_at'] = datetime.now().isoformat()
#                 break
        
#         self._save_json(self.reminders_file, reminders)
        
#         return {
#             "success": True,
#             "message": f"Reminder {reminder_id} marked as triggered"
#         }
    
#     def delete_reminder(self, reminder_id: int) -> Dict:
#         """Delete a reminder"""
#         reminders = self._load_json(self.reminders_file)
#         original_length = len(reminders)
        
#         reminders = [r for r in reminders if r['id'] != reminder_id]
        
#         if len(reminders) == original_length:
#             return {
#                 "success": False,
#                 "message": f"Reminder {reminder_id} not found"
#             }
        
#         self._save_json(self.reminders_file, reminders)
        
#         return {
#             "success": True,
#             "message": f"Reminder {reminder_id} deleted"
#         }
    
#     # ==================== NOTIFICATIONS LOG ====================
    
#     def _log_notification(self, notification_type: str, recipient: str,
#                          subject: str, message: str, success: bool,
#                          error: Optional[str] = None):
#         """Log notification attempt"""
#         notifications = self._load_json(self.notifications_file)
        
#         notification = {
#             "id": len(notifications) + 1,
#             "type": notification_type,
#             "recipient": recipient,
#             "subject": subject,
#             "message": message[:100] + "..." if len(message) > 100 else message,
#             "success": success,
#             "error": error,
#             "timestamp": datetime.now().isoformat()
#         }
        
#         notifications.append(notification)
        
#         # Keep only last 1000 notifications
#         if len(notifications) > 1000:
#             notifications = notifications[-1000:]
        
#         self._save_json(self.notifications_file, notifications)
    
#     def get_notification_history(self, limit: int = 50) -> List[Dict]:
#         """Get notification history"""
#         notifications = self._load_json(self.notifications_file)
#         return notifications[-limit:]
    
#     def get_email_config_status(self) -> Dict:
#         """Get email configuration status"""
#         return {
#             "email_enabled": self.email_enabled,
#             "smtp_server": self.smtp_server,
#             "smtp_port": self.smtp_port,
#             "email_address": self.email_address if self.email_enabled else "Not configured",
#             "configuration_help": "Set EMAIL_ADDRESS and EMAIL_PASSWORD environment variables. For Gmail, use App Password.",
#             "app_password_link": "https://support.google.com/accounts/answer/185833"
#         }




import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import os


class NotificationService:
    """
    Cloud-Safe Notification + Reminder Service
    Works reliably on Render, HuggingFace, Docker, and any ephemeral filesystem.
    Uses /tmp for storage (always writable).
    """

    def __init__(self):
        # Cloud-SAFE storage directory
        self.data_dir = Path("/tmp/notifications")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Files inside /tmp (safe for HuggingFace + Render)
        self.reminders_file = self.data_dir / "reminders.json"
        self.notifications_file = self.data_dir / "notifications.json"

        # Email configuration (from environment variables)
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_address = os.getenv("EMAIL_ADDRESS", "")
        self.email_password = os.getenv("EMAIL_PASSWORD", "")

        # Email enabled only if both env vars exist
        self.email_enabled = bool(self.email_address and self.email_password)

        self._init_files()

    # ==================================================================
    # INTERNAL HELPERS
    # ==================================================================

    def _init_files(self):
        """Initialize JSON files in /tmp"""
        if not self.reminders_file.exists():
            self._save_json(self.reminders_file, [])

        if not self.notifications_file.exists():
            self._save_json(self.notifications_file, [])

    def _load_json(self, filepath: Path):
        """Load JSON safely"""
        try:
            if not filepath.exists():
                return []
            with open(filepath, "r") as f:
                return json.load(f)
        except:
            return []

    def _save_json(self, filepath: Path, data: any):
        """Save JSON safely in /tmp"""
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Could not save JSON {filepath}: {e}")

    # ==================================================================
    # EMAIL HANDLING
    # ==================================================================

    def send_email(self, to_email: str, subject: str, body: str,
                   body_html: Optional[str] = None) -> Dict:
        """
        Send an email using SMTP.
        Works only if EMAIL_ADDRESS + EMAIL_PASSWORD are set.
        """

        if not self.email_enabled:
            return {
                "success": False,
                "message": "Email is not configured. Set EMAIL_ADDRESS & EMAIL_PASSWORD."
            }

        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = self.email_address
            msg["To"] = to_email
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "plain"))

            if body_html:
                msg.attach(MIMEText(body_html, "html"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_address, self.email_password)
                server.send_message(msg)

            self._log_notification("EMAIL", to_email, subject, body, True)

            return {
                "success": True,
                "message": f"Email sent successfully to {to_email}"
            }

        except Exception as e:
            self._log_notification("EMAIL", to_email, subject, body, False, str(e))
            return {
                "success": False,
                "message": f"Failed to send email: {str(e)}"
            }

    def send_price_alert_email(self, to_email: str, ticker: str,
                               current_price: float, target_price: float,
                               condition: str) -> Dict:
        """Send price alert email"""

        subject = f"🚨 Price Alert: {ticker}"

        body = (
            f"Price alert for {ticker}\n\n"
            f"Current Price: ₹{current_price}\n"
            f"Target Price: ₹{target_price}\n"
            f"Condition: {condition}\n\n"
            f"Triggered at: {datetime.now()}\n"
        )

        body_html = f"""
        <html><body>
        <h3>🚨 Price Alert: {ticker}</h3>
        <p><b>Current Price:</b> ₹{current_price}</p>
        <p><b>Target Price:</b> ₹{target_price}</p>
        <p><b>Condition:</b> {condition}</p>
        <p>Triggered at: {datetime.now()}</p>
        </body></html>
        """

        return self.send_email(to_email, subject, body, body_html)

    def send_portfolio_summary_email(self, to_email: str, data: Dict) -> Dict:
        """Send formatted portfolio summary email"""
        subject = "📊 Your Portfolio Summary"

        body = (
            f"Total Investment: ₹{data['total_investment']}\n"
            f"Current Value: ₹{data['total_current_value']}\n"
            f"P&L: ₹{data['total_profit_loss']} ({data['total_profit_loss_pct']}%)\n"
        )

        holdings_html = ""
        for h in data["holdings"]:
            color = "#4caf50" if h["profit_loss"] >= 0 else "#f44336"
            holdings_html += f"""
            <tr>
                <td>{h['ticker']}</td>
                <td>{h['quantity']}</td>
                <td>₹{h['avg_buy_price']}</td>
                <td>₹{h['current_price']}</td>
                <td style="color:{color};">₹{h['profit_loss']} ({h['profit_loss_pct']}%)</td>
            </tr>
            """

        body_html = f"""
        <html><body>
        <h2>📊 Portfolio Summary</h2>
        <table border="1" cellspacing="0" cellpadding="6">
            <tr><th>Stock</th><th>Qty</th><th>Avg Buy</th><th>Current</th><th>P&L</th></tr>
            {holdings_html}
        </table>
        </body></html>
        """

        return self.send_email(to_email, subject, body, body_html)

    # ==================================================================
    # REMINDERS (Cloud-Safe)
    # ==================================================================

    def create_reminder(self, title: str, message: str,
                        reminder_time: str, reminder_type: str = "GENERAL",
                        ticker: Optional[str] = None) -> Dict:

        reminders = self._load_json(self.reminders_file)

        reminder = {
            "id": (reminders[-1]["id"] + 1) if reminders else 1,
            "title": title,
            "message": message,
            "reminder_time": reminder_time,
            "type": reminder_type,
            "ticker": ticker,
            "created_at": datetime.now().isoformat(),
            "status": "ACTIVE",
            "triggered": False
        }

        reminders.append(reminder)
        self._save_json(self.reminders_file, reminders)

        return {"success": True, "reminder": reminder}

    def get_reminders(self, status: Optional[str] = None):
        reminders = self._load_json(self.reminders_file)
        if status:
            return [r for r in reminders if r["status"] == status.upper()]
        return reminders

    def mark_reminder_triggered(self, reminder_id: int):
        reminders = self._load_json(self.reminders_file)

        for r in reminders:
            if r["id"] == reminder_id:
                r["triggered"] = True
                r["triggered_at"] = datetime.now().isoformat()

        self._save_json(self.reminders_file, reminders)
        return {"success": True}

    # ==================================================================
    # NOTIFICATION LOGS
    # ==================================================================

    def _log_notification(self, ntype: str, recipient: str,
                          subject: str, message: str, success: bool,
                          error: Optional[str] = None):

        logs = self._load_json(self.notifications_file)

        entry = {
            "id": (logs[-1]["id"] + 1) if logs else 1,
            "type": ntype,
            "recipient": recipient,
            "subject": subject,
            "message": message[:100] + "...",
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }

        logs.append(entry)

        # Prevent huge file growth
        logs = logs[-500:]

        self._save_json(self.notifications_file, logs)

    def get_notification_history(self, limit: int = 50):
        logs = self._load_json(self.notifications_file)
        return logs[-limit:]

    def get_email_config_status(self):
        return {
            "email_enabled": self.email_enabled,
            "smtp_server": self.smtp_server,
            "smtp_port": self.smtp_port,
            "email_address": self.email_address if self.email_enabled else "Not Configured",
            "help": "Set EMAIL_ADDRESS and EMAIL_PASSWORD env vars. Gmail requires an App Password."
        }

