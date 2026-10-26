"""AWS cloud layer: S3 (snapshot), DynamoDB (incidents), Bedrock (risk summary), SES (email).

Every call degrades gracefully: if a service is not configured the app keeps
working locally and logs to incidents_local.jsonl.
"""
import html
import json
import os
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone

import boto3
from botocore.config import Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RULE_BASED_RISK = {
    "FALL_SUSPECTED": "HIGH",
    "VOICE_DISTRESS": "HIGH",
    "IMMOBILE": "MEDIUM",
    "FACE_LOST": "MEDIUM",
}


class CloudClient:
    def __init__(self):
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.bucket = os.getenv("S3_BUCKET", "")
        self.table_name = os.getenv("DDB_TABLE", "ChronosIncidents")
        self.api_base = os.getenv("API_BASE_URL", "").rstrip("/")
        self.model_id = os.getenv("BEDROCK_MODEL_ID", "")
        self.bedrock_region = os.getenv("BEDROCK_REGION", self.region)
        self.email_backend = os.getenv("EMAIL_BACKEND", "ses").lower()   # "ses" or "smtp"
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_pass = os.getenv("SMTP_APP_PASSWORD", "").replace(" ", "")
        self.smtp_ready = bool(self.smtp_user and self.smtp_pass)

        self.store = os.getenv("STORE", "s3" if self.bucket else "dynamodb").lower()  # s3 or dynamodb
        self.ses = self.s3 = self.table = self.bedrock = self.logs = None
        self.log_group = os.getenv("CW_LOG_GROUP", "/chronos/agent")
        self.log_stream = None
        self._logs_ok = True
        self._log_lock = threading.Lock()
        self.local_mode = False
        try:
            if boto3.Session().get_credentials() is None:
                self.local_mode = True
        except Exception:
            self.local_mode = True
        if self.local_mode:
            print("No AWS credentials found: running in LOCAL mode (no cloud calls).")
            return
        try:
            self.ses = boto3.client("ses", region_name=os.getenv("SES_REGION", self.region))
            if self.bucket:
                self.s3 = boto3.client("s3", region_name=self.region,
                                       config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
            if self.store == "dynamodb" and self.table_name:
                self.table = boto3.resource("dynamodb", region_name=self.region).Table(self.table_name)
            if self.model_id:
                self.bedrock = boto3.client("bedrock-runtime", region_name=self.bedrock_region)
            if os.getenv("CLOUDWATCH", "on").lower() != "off":
                self.logs = boto3.client("logs", region_name=self.region)
        except Exception as e:
            print("AWS client init problem:", e)

    def describe(self):
        def mark(ok):
            return "ON " if ok else "OFF"
        return "\n".join([
            f"  SES email ........ {mark(self.ses and self.email_backend != 'smtp')}",
            f"  Gmail SMTP backup  {mark(self.smtp_ready)}  ({self.smtp_user or 'SMTP_USER not set'})",
            f"  S3 snapshots ..... {mark(self.s3 and self.bucket)}  ({self.bucket or 'S3_BUCKET not set'})",
            f"  Incident store ... {self.store.upper()}  ({'S3 bucket ' + self.bucket if self.store == 's3' else self.table_name})",
            f"  Bedrock summary .. {mark(self.bedrock)}  ({self.model_id or 'BEDROCK_MODEL_ID not set'})",
            f"  Review web page .. {mark(self.api_base)}  ({self.api_base or 'API_BASE_URL not set'})",
            f"  CloudWatch logs .. {mark(self.logs)}  ({self.log_group})",
        ])

    @property
    def can_poll(self):
        return self.table is not None or (self.store == "s3" and self.s3 is not None and bool(self.bucket))

    # ---------------- CloudWatch Logs (observability) ----------------
    def log_event(self, record):
        """Send one agent trace record to CloudWatch Logs. Silent on failure and disables itself."""
        if not self.logs or not self._logs_ok:
            return
        with self._log_lock:
            try:
                if self.log_stream is None:
                    try:
                        self.logs.create_log_group(logGroupName=self.log_group)
                    except Exception as e:
                        if "ResourceAlreadyExists" not in str(e):
                            raise
                    self.log_stream = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                    self.logs.create_log_stream(logGroupName=self.log_group, logStreamName=self.log_stream)
                self.logs.put_log_events(
                    logGroupName=self.log_group, logStreamName=self.log_stream,
                    logEvents=[{"timestamp": int(time.time() * 1000),
                                "message": json.dumps(record, ensure_ascii=False)}])
            except Exception as e:
                print("[CloudWatch] logging disabled:", str(e)[:160])
                self._logs_ok = False

    # ---------------- Bedrock ----------------
    def summarize(self, ctx):
        """Returns (risk, summary_text). Uses Bedrock if configured, else rules."""
        fallback_risk = RULE_BASED_RISK.get(ctx["kind"], "MEDIUM")
        if ctx.get("no_response") and fallback_risk == "MEDIUM":
            fallback_risk = "HIGH"
        fallback = (f"{ctx['kind']} detected ({ctx['evidence']}). "
                    f"Patient did not confirm safety. Risk assessed as {fallback_risk} by rules.")
        if not self.bedrock:
            return fallback_risk, fallback

        prompt = (
            "You assist a physician. An elderly-care camera robot raised an alert.\n"
            f"Patient: {ctx['age']} year old {ctx['gender']}, condition: {ctx['condition']}.\n"
            f"Alert type: {ctx['kind']}. Vision evidence: {ctx['evidence']}.\n"
            f"Check-in events (JSON): {json.dumps(ctx['events'])[:1500]}\n"
            "Reply in exactly this format:\n"
            "RISK: LOW or MEDIUM or HIGH\n"
            "SUMMARY: two short sentences for the physician, plain language, no speculation."
        )
        try:
            resp = self.bedrock.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 250, "temperature": 0.2},
            )
            text = resp["output"]["message"]["content"][0]["text"].strip()
            risk, summary = fallback_risk, text
            for line in text.splitlines():
                if line.upper().startswith("RISK:"):
                    val = line.split(":", 1)[1].strip().upper()
                    if val in ("LOW", "MEDIUM", "HIGH"):
                        risk = val
                elif line.upper().startswith("SUMMARY:"):
                    summary = line.split(":", 1)[1].strip()
            return risk, summary
        except Exception as e:
            print("Bedrock error:", e)
            return fallback_risk, fallback

    # ---------------- Incident store (S3 or DynamoDB) ----------------
    def _presign(self, method, key, expires=259200, **extra):
        params = {"Bucket": self.bucket, "Key": key}
        params.update(extra)
        return self.s3.generate_presigned_url(method, Params=params, ExpiresIn=expires)

    def _review_page(self, item, snapshot_url, dispatch_url, false_url):
        e = html.escape
        js_urls = json.dumps({"dispatch": dispatch_url, "false": false_url}).replace("</", "<\\/")
        img = f'<img src="{e(snapshot_url)}" alt="Snapshot" style="max-width:100%;border-radius:6px">' if snapshot_url \
            else "<p><i>No snapshot stored.</i></p>"
        color = {"HIGH": "#d9534f", "MEDIUM": "#c77700", "LOW": "#2e8b57"}.get(item["risk"], "#333")
        return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Chronos Care incident</title>
<style>body{{font-family:Arial,sans-serif;max-width:720px;margin:24px auto;padding:0 16px;line-height:1.5}}
.card{{border:1px solid #ddd;border-radius:8px;padding:16px;margin:12px 0}}
button{{padding:12px 22px;border:0;border-radius:6px;color:#fff;font-weight:bold;font-size:15px;cursor:pointer;margin-right:8px}}</style></head>
<body><h2>Chronos Care incident</h2>
<div class="card">
<p><b>Risk:</b> <span style="color:{color};font-weight:bold">{e(item['risk'])}</span></p>
<p><b>Patient:</b> {e(item['patient_age'])} y/o {e(item['patient_gender'])}, {e(item['condition'])}</p>
<p><b>Alert:</b> {e(item['kind'])}<br><b>Vision evidence:</b> {e(item['evidence'])}</p>
<p><b>AI summary:</b> {e(item['summary'])}</p>
<p><small>{e(item['created_at'])} &middot; {e(item['incident_id'])}</small></p>
{img}
</div>
<div id="btns">
<button style="background:#d9534f" onclick="decide('dispatch','DISPATCHED')">DISPATCH MEDICAL TEAM</button>
<button style="background:#5cb85c" onclick="decide('false','FALSE ALARM')">FALSE ALARM</button>
</div>
<p id="msg"></p>
<script>
const URLS = {js_urls};
async function decide(which, label) {{
  document.getElementById('btns').style.display = 'none';
  const msg = document.getElementById('msg');
  msg.textContent = 'Sending...';
  try {{
    const r = await fetch(URLS[which], {{method: 'PUT', body: label}});
    msg.textContent = r.ok ? 'Decision recorded: ' + label + '. The robot will respond within seconds.'
                           : 'Could not record the decision (HTTP ' + r.status + '). Reload and try again.';
    if (!r.ok) document.getElementById('btns').style.display = 'block';
  }} catch (err) {{
    msg.textContent = 'Network error. Reload and try again.';
    document.getElementById('btns').style.display = 'block';
  }}
}}
</script></body></html>"""

    def create_incident(self, ctx, snapshot_jpg, risk, summary):
        incident_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        token = secrets.token_urlsafe(16)
        snapshot_key = ""
        if self.s3 and self.bucket and snapshot_jpg:
            try:
                snapshot_key = f"snapshots/{incident_id}.jpg"
                self.s3.put_object(Bucket=self.bucket, Key=snapshot_key, Body=snapshot_jpg,
                                   ContentType="image/jpeg", ServerSideEncryption="AES256")
            except Exception as e:
                print("S3 upload error:", e)
                snapshot_key = ""

        item = {
            "incident_id": incident_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "PENDING",
            "token": token,
            "kind": ctx["kind"],
            "risk": risk,
            "evidence": str(ctx["evidence"]),
            "patient_age": str(ctx["age"]),
            "patient_gender": str(ctx["gender"]),
            "condition": str(ctx["condition"]),
            "summary": summary,
            "snapshot_key": snapshot_key,
        }
        safe = {k: v for k, v in item.items() if k != "token"}
        review_url = ""

        if self.store == "s3" and self.s3 and self.bucket:
            try:
                self.s3.put_object(Bucket=self.bucket, Key=f"incidents/{incident_id}.json",
                                   Body=json.dumps(safe, ensure_ascii=False).encode("utf-8"),
                                   ContentType="application/json", ServerSideEncryption="AES256")
                snap_url = self._presign("get_object", snapshot_key) if snapshot_key else ""
                page = self._review_page(
                    item, snap_url,
                    self._presign("put_object", f"decisions/{incident_id}/DISPATCHED"),
                    self._presign("put_object", f"decisions/{incident_id}/FALSE_ALARM"))
                self.s3.put_object(Bucket=self.bucket, Key=f"review/{incident_id}.html",
                                   Body=page.encode("utf-8"), ContentType="text/html; charset=utf-8",
                                   ServerSideEncryption="AES256")
                review_url = self._presign("get_object", f"review/{incident_id}.html",
                                           ResponseContentType="text/html; charset=utf-8")
            except Exception as e:
                print("S3 incident store error:", e)
        else:
            if self.table:
                try:
                    self.table.put_item(Item=item)
                except Exception as e:
                    print("DynamoDB put error:", e)
            if self.api_base:
                review_url = f"{self.api_base}/incident/{incident_id}?token={token}"

        try:
            with open(os.path.join(BASE_DIR, "incidents_local.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(safe, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return {"incident_id": incident_id, "token": token, "review_url": review_url,
                "snapshot_key": snapshot_key}

    def get_status(self, incident_id):
        if self.store == "s3":
            if not (self.s3 and self.bucket):
                return None
            try:
                resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=f"decisions/{incident_id}/")
                names = [o["Key"].rsplit("/", 1)[-1] for o in resp.get("Contents", [])]
                if "DISPATCHED" in names:
                    return "DISPATCHED"
                if "FALSE_ALARM" in names:
                    return "FALSE_ALARM"
                return "PENDING"
            except Exception as e:
                print("S3 status error:", e)
                return None
        if not self.table:
            return None
        try:
            item = self.table.get_item(Key={"incident_id": incident_id}).get("Item")
            return item.get("status") if item else None
        except Exception as e:
            print("DynamoDB get error:", e)
            return None

    def set_status(self, incident_id, status, by):
        if self.store == "s3":
            if self.s3 and self.bucket:
                try:
                    self.s3.put_object(Bucket=self.bucket, Key=f"decisions/{incident_id}/{status}",
                                       Body=by.encode("utf-8"), ServerSideEncryption="AES256")
                except Exception as e:
                    print("S3 status write error:", e)
            return
        if not self.table:
            return
        try:
            self.table.update_item(
                Key={"incident_id": incident_id},
                UpdateExpression="SET #s=:s, decided_by=:b, decided_at=:t",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":s": status, ":b": by,
                                           ":t": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            )
        except Exception as e:
            print("DynamoDB update error:", e)

    # ---------------- SES ----------------
    def send_email(self, receiver, sender, incident, ctx, risk, summary, reminder=False):
        if not self.ses and not self.smtp_ready:
            print("[EMAIL] no email backend configured; email skipped.")
            return False
        color = {"HIGH": "#d9534f", "MEDIUM": "#f0ad4e", "LOW": "#5cb85c"}.get(risk, "#d9534f")
        e = lambda v: html.escape(str(v))
        button = (f'<p><a href="{e(incident["review_url"])}" style="background:{color};color:#fff;'
                  'padding:12px 25px;text-decoration:none;border-radius:5px;font-weight:bold;">'
                  'REVIEW &amp; DECIDE</a></p>') if incident["review_url"] else \
                 "<p><i>Review link not configured (API_BASE_URL).</i></p>"
        prefix = "REMINDER - NOT ACKNOWLEDGED: " if reminder else ""
        body = f"""
        <html><body style="font-family:Arial,sans-serif;padding:20px;line-height:1.6;">
          <h2 style="color:{color};">{e(prefix)}Chronos Care alert - risk {e(risk)}</h2>
          <p><b>Patient:</b> {e(ctx['age'])} y/o {e(ctx['gender'])} &middot; <b>Condition:</b> {e(ctx['condition'])}</p>
          <p><b>Alert type:</b> {e(ctx['kind'])}<br><b>Vision evidence:</b> {e(ctx['evidence'])}</p>
          <p><b>AI summary:</b> {e(summary)}</p>
          <p><b>Incident:</b> {e(incident['incident_id'])}</p>
          {button}
          <p style="color:#888;font-size:12px;">You can dispatch a medical team or mark a false alarm
          on the review page. The snapshot is visible only there and expires automatically.</p>
        </body></html>"""
        subject = f"{prefix}[{risk}] Patient alert: {ctx['kind']} ({ctx['age']} y/o)"
        text = (f"{prefix}Chronos alert ({risk}): {ctx['kind']}. {summary} "
                f"Review: {incident['review_url'] or 'not configured'}")

        if self.ses and self.email_backend != "smtp":
            try:
                self.ses.send_email(
                    Source=sender,
                    Destination={"ToAddresses": [receiver]},
                    Message={
                        "Subject": {"Data": subject, "Charset": "UTF-8"},
                        "Body": {"Html": {"Data": body, "Charset": "UTF-8"},
                                 "Text": {"Data": text, "Charset": "UTF-8"}},
                    },
                )
                print("[SES] email sent.")
                return True
            except Exception as ex:
                print("[SES] error:", str(ex)[:200])
        if self.smtp_ready:
            return self._send_smtp(receiver, subject, text, body)
        return False

    def _send_smtp(self, receiver, subject, text, html_body):
        import smtplib
        import ssl
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.smtp_user
        msg["To"] = receiver
        msg.set_content(text)
        msg.add_alternative(html_body, subtype="html")
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=30) as smtp:
                smtp.login(self.smtp_user, self.smtp_pass)
                smtp.send_message(msg)
            print("[SMTP] email sent via Gmail.")
            return True
        except Exception as ex:
            print("[SMTP] error:", ex)
            return False
