"""
Notification Service - Multi-channel alert delivery
Sends alerts to console, webhooks (Slack/Discord), email, and backend API
"""
import logging
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional
from datetime import datetime
from .config import settings

logger = logging.getLogger("notification-service")


class NotificationService:
    """
    Multi-channel notification service for medical alerts
    """
    
    def __init__(self):
        self.alert_history = []
        logger.info("📬 Notification Service initialized")
        logger.info(f"   Console alerts: {settings.ENABLE_CONSOLE_ALERTS}")
        logger.info(f"   Webhook alerts: {settings.ENABLE_WEBHOOK_ALERTS}")
        logger.info(f"   Email alerts: {settings.ENABLE_EMAIL_ALERTS}")
        logger.info(f"   Email alerts: {settings.ENABLE_EMAIL_ALERTS}")
        logger.info(f"   Email alerts: {settings.ENABLE_EMAIL_ALERTS}")
        if settings.ENABLE_EMAIL_ALERTS:
            logger.info(f"   Email recipients: {settings.ALERT_EMAIL_TO}")
    
    def send_alert(self, alert: Dict):
        """
        Send alert through all enabled channels
        
        Args:
            alert: Alert dictionary from LangChain agent
        """
        try:
            # Store in history
            self.alert_history.append(alert)
            
            # Keep only last 100 alerts
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
            
            # Send to console (always for demo visibility)
            if settings.ENABLE_CONSOLE_ALERTS:
                self._send_console_alert(alert)
            
            # Send to webhook (Slack/Discord)
            if settings.ENABLE_WEBHOOK_ALERTS and settings.WEBHOOK_URL:
                self._send_webhook_alert(alert)
            
            # Send to email
            if settings.ENABLE_EMAIL_ALERTS and settings.ALERT_EMAIL_TO:
                self._send_email_alert(alert)
            
            # Log to backend API (for dashboard display)
            self._log_to_backend(alert)
            
            logger.info(f"✅ Alert sent for patient {alert['patient_id']}")
        
        except Exception as e:
            logger.error(f"❌ Failed to send alert: {e}")
    
    def _send_console_alert(self, alert: Dict):
        """Display alert in console with formatting"""
        severity = alert['severity']
        patient_id = alert['patient_id']
        floor_id = alert['floor_id']
        message = alert['alert_message']
        
        # Color-coded severity indicators
        emoji_map = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡"
        }
        emoji = emoji_map.get(severity, "⚠️")
        
        # Format output
        print("\n" + "=" * 80)
        print(f"{emoji} {severity} ALERT - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 80)
        print(f"Patient: {patient_id} | Floor: {floor_id}")
        print("-" * 80)
        print(message)
        print("-" * 80)
        print(f"LLM Provider: {alert.get('llm_provider', 'unknown').upper()}")
        print("=" * 80 + "\n")
    
    def _send_webhook_alert(self, alert: Dict):
        """Send alert to Slack/Discord webhook"""
        try:
            patient_id = alert['patient_id']
            severity = alert['severity']
            floor_id = alert['floor_id']
            message = alert['alert_message']
            
            # Format for Slack/Discord
            severity_emoji = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MEDIUM": "🟡"
            }
            
            # Slack/Discord compatible payload
            payload = {
                "text": f"{severity_emoji.get(severity, '⚠️')} **{severity} ALERT**",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"🚨 {severity} Alert - ICU Monitoring System"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Patient:*\n{patient_id}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Floor:*\n{floor_id}"
                            }
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"```{message}```"
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"Generated by LangChain ({alert.get('llm_provider', 'unknown')}) | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                            }
                        ]
                    }
                ]
            }
            
            # Send to webhook
            response = requests.post(
                settings.WEBHOOK_URL,
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Webhook alert sent for {patient_id}")
            else:
                logger.warning(f"⚠️ Webhook returned status {response.status_code}")
        
        except Exception as e:
            logger.error(f"❌ Webhook alert failed: {e}")
    
    def _send_email_alert(self, alert: Dict):
        """Send alert via email to doctor"""
        try:
            patient_id = alert['patient_id']
            severity = alert['severity']
            floor_id = alert['floor_id']
            message_body = alert['alert_message']
            timestamp = alert.get('generated_at', datetime.utcnow().isoformat())
            
            # Email subject
            subject = f"🚨 {severity} ALERT - Patient {patient_id} ({floor_id})"
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = settings.ALERT_EMAIL_FROM
            msg['To'] = settings.ALERT_EMAIL_TO
            
            # Plain text version
            text_content = f"""
ICU EMERGENCY ALERT
{'='*60}

Severity: {severity}
Patient: {patient_id}
Floor: {floor_id}
Time: {timestamp}

{'='*60}

{message_body}

{'='*60}

This is an automated alert from the ICU Monitoring System.
Powered by LangChain + Gemini AI.

DO NOT REPLY TO THIS EMAIL.
"""
            
            # HTML version (more professional)
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #{'dc3545' if severity == 'CRITICAL' else 'ff9800' if severity == 'HIGH' else 'ffc107'}; 
                   color: white; padding: 20px; text-align: center; border-radius: 5px; }}
        .content {{ background-color: #f8f9fa; padding: 20px; margin-top: 20px; border-radius: 5px; }}
        .info-box {{ background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }}
        .footer {{ text-align: center; margin-top: 20px; color: #6c757d; font-size: 12px; }}
        pre {{ white-space: pre-wrap; word-wrap: break-word; background-color: #f4f4f4; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚨 {severity} ALERT</h1>
            <p>ICU Monitoring System</p>
        </div>
        
        <div class="content">
            <div class="info-box">
                <p><strong>Patient:</strong> {patient_id}</p>
                <p><strong>Floor:</strong> {floor_id}</p>
                <p><strong>Time:</strong> {timestamp}</p>
                <p><strong>Severity:</strong> {severity}</p>
            </div>
            
            <h3>Alert Details:</h3>
            <pre>{message_body}</pre>
        </div>
        
        <div class="footer">
            <p>This is an automated alert from the ICU Monitoring System.</p>
            <p>Powered by LangChain + Gemini AI (gemini-2.5-flash)</p>
            <p><em>DO NOT REPLY TO THIS EMAIL</em></p>
        </div>
    </div>
</body>
</html>
"""
            
            # Attach parts
            part1 = MIMEText(text_content, 'plain')
            part2 = MIMEText(html_content, 'html')
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
                server.starttls()  # Secure connection
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                
                # Support multiple recipients (comma-separated)
                recipients = [email.strip() for email in settings.ALERT_EMAIL_TO.split(',')]
                server.sendmail(settings.ALERT_EMAIL_FROM, recipients, msg.as_string())
            
            logger.info(f"✅ Email alert sent to {settings.ALERT_EMAIL_TO} for patient {patient_id}")
        
        except Exception as e:
            logger.error(f"❌ Email alert failed: {e}")
    
    def _log_to_backend(self, alert: Dict):
        """Log alert to backend API (for future dashboard display)"""
        try:
            # This endpoint doesn't exist yet, but we'll log it
            # In the future, you can create POST /api/alerts endpoint
            endpoint = f"{settings.BACKEND_API_URL}/api/alerts"
            
            # For now, just log locally
            logger.debug(f"📊 Alert logged: {alert['patient_id']} - {alert['severity']}")
            
            # Uncomment when backend endpoint is ready:
            # response = requests.post(endpoint, json=alert, timeout=3)
            # logger.debug(f"Backend API response: {response.status_code}")
        
        except Exception as e:
            logger.debug(f"Backend logging failed (expected during development): {e}")
    
    def get_alert_history(self, limit: int = 10) -> list:
        """Get recent alert history"""
        return self.alert_history[-limit:]
    
    def get_alert_statistics(self) -> Dict:
        """Get alert statistics"""
        if not self.alert_history:
            return {
                "total_alerts": 0,
                "by_severity": {},
                "by_floor": {},
                "by_patient": {}
            }
        
        by_severity = {}
        by_floor = {}
        by_patient = {}
        
        for alert in self.alert_history:
            severity = alert.get('severity', 'UNKNOWN')
            floor = alert.get('floor_id', 'UNKNOWN')
            patient = alert.get('patient_id', 'UNKNOWN')
            
            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_floor[floor] = by_floor.get(floor, 0) + 1
            by_patient[patient] = by_patient.get(patient, 0) + 1
        
        return {
            "total_alerts": len(self.alert_history),
            "by_severity": by_severity,
            "by_floor": by_floor,
            "by_patient": by_patient,
            "last_alert_time": self.alert_history[-1].get('generated_at') if self.alert_history else None
        }
