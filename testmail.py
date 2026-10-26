
import os
import smtplib
import ssl
from email.message import EmailMessage

from dotenv import load_dotenv

load_dotenv()
user = os.getenv("SMTP_USER", "").strip()
pw = os.getenv("SMTP_APP_PASSWORD", "").replace(" ", "").strip()

print("EMAIL_BACKEND :", os.getenv("EMAIL_BACKEND"))
print("SMTP_USER     :", user if user else "NOT SET")
print("Password length:", len(pw), "(a Gmail App password has exactly 16 characters)")

if not user or not pw:
    raise SystemExit("\nSMTP_USER or SMTP_APP_PASSWORD is missing in .env")

msg = EmailMessage()
msg["Subject"] = "Chronos email test"
msg["From"] = user
msg["To"] = user
msg.set_content("If you can read this, Gmail sending works.")

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=30) as smtp:
        smtp.login(user, pw)
        smtp.send_message(msg)
    print("\nSUCCESS: test email sent to", user, "- check Inbox and Spam.")
except smtplib.SMTPAuthenticationError as e:
    print("\nLOGIN FAILED:", e.smtp_code, "")
    print("Most likely causes:")
    print(" 1) 2-Step Verification is not ON for this Google account")
    print(" 2) The App password was copied wrongly (must be 16 letters, no normal password)")
    print(" 3) The App password was created in a different Google account than SMTP_USER")
except Exception as e:
    print("\nERROR:", type(e).__name__, e)
    print("If this says timeout / connection: check internet, VPN, or firewall blocking port 465.")