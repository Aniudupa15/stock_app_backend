"""Send a WhatsApp message through the personal alert Lambda (deploy/whatsapp_lambda).

Config (repo-root .env):
    WHATSAPP_NOTIFY_URL     = https://<id>.lambda-url.ap-south-1.on.aws/
    WHATSAPP_NOTIFY_SECRET  = <from deploy/whatsapp_lambda/.deploy.env>
    WHATSAPP_TO             = me   (default; or 91XXXXXXXXXX, or a group JID like 1203...@g.us;
                                    comma-separated for several recipients - used by notify())

Usage:
    from scripts.whatsapp_notify import notify, send_whatsapp
    notify("Momentum picks report", body)   # every WHATSAPP_TO recipient, long bodies split into parts
    send_whatsapp("Stop-loss hit: INFY")    # one message, one recipient
    python scripts/whatsapp_notify.py "hello from the CLI"

If the URL/secret are absent it prints the message instead (dry run), like the email scripts.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MAX_CHARS = 4000  # Lambda rejects longer; WhatsApp itself allows more but long texts are unreadable on phone.


def send_whatsapp(message: str, to: str | None = None, retries: int = 3) -> bool:
    """Returns True if WhatsApp accepted the message. Never raises - alerting must not crash the caller."""
    url = os.environ.get("WHATSAPP_NOTIFY_URL")
    secret = os.environ.get("WHATSAPP_NOTIFY_SECRET")
    to = to or os.environ.get("WHATSAPP_TO", "me")
    if len(message) > MAX_CHARS:
        message = message[: MAX_CHARS - 20] + "\n...(truncated)"
    if not url or not secret:
        print("[dry run: WHATSAPP_NOTIFY_URL/SECRET not set - printing message]\n")
        print(message)
        return False

    payload = json.dumps({"to": to, "message": message}).encode()
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"content-type": "application/json", "x-api-key": secret},
        )
        try:
            with urllib.request.urlopen(req, timeout=100) as resp:
                print(f"WhatsApp sent to {to}")
                return json.loads(resp.read() or b"{}").get("success", False)
        except urllib.error.HTTPError as exc:
            # ascii() keeps mudslide's emoji log symbols from crashing a cp1252 Windows console.
            detail = exc.read().decode(errors="replace")[:300].encode("ascii", "replace").decode()
            # 429 = another send is holding the single concurrency slot; 5xx = transient connect failure.
            retryable = exc.code == 429 or (exc.code >= 500 and "logged_out" not in detail)
            print(f"WhatsApp send failed (HTTP {exc.code}, attempt {attempt}): {detail}")
            if not retryable:
                return False
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"WhatsApp send failed (attempt {attempt}): {exc}")
        if attempt < retries:
            time.sleep(10 * attempt)
    return False


def _chunks(text: str, limit: int) -> list[str]:
    """Split on line boundaries so tables in the reports never break mid-row."""
    parts, cur = [], ""
    for line in text.splitlines():
        while len(line) > limit:  # a single monster line - hard split
            parts.append(line[:limit])
            line = line[limit:]
        if cur and len(cur) + 1 + len(line) > limit:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return parts


def notify(subject: str, body: str) -> None:
    """Send a report to every WHATSAPP_TO recipient: bold subject, then the body, split into numbered
    parts if it is longer than one message. Sends are sequential so the Lambda's lock is never contended."""
    recipients = [r.strip() for r in os.environ.get("WHATSAPP_TO", "me").split(",") if r.strip()]
    parts = _chunks(f"*{subject}*\n\n{body}", MAX_CHARS - 20)
    for to in recipients:
        for i, part in enumerate(parts, 1):
            prefix = f"({i}/{len(parts)}) " if len(parts) > 1 else ""
            if not send_whatsapp(prefix + part, to=to):
                break  # don't send part 3 if part 2 failed - the reader would get a gap


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "Test message from stock_app_backend"
    sys.exit(0 if send_whatsapp(text) else 1)
