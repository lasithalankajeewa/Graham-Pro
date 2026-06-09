import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL") or SMTP_USER


def send_alert_email(to_email, subject, body):
    if not SMTP_USER or not SMTP_PASSWORD:
        return False, "SMTP not configured"
    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(FROM_EMAIL, to_email, msg.as_string())
        return True, "Sent"
    except Exception as e:
        return False, str(e)


def send_test_email(to_email):
    return send_alert_email(
        to_email,
        "Graham-Bot — Test Email",
        "Your Graham-Bot email alerts are configured correctly.\n\n---\nGraham-Bot | For educational purposes only.",
    )
