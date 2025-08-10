# EscalateAIV610082025.py
# Full corrected + dedupe + compact Kanban + Twilio WhatsApp (optional)
# Preserves UI labels and features from the uploaded file.

import os
import re
import time
import hashlib
import threading
import datetime
import sqlite3
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import numpy as np
import imaplib
import email
from email.header import decode_header
from email.message import EmailMessage
import smtplib
import requests

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------
# Load env once (centralized)
# -----------------------
load_dotenv()

EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
ALERT_RECIPIENT = os.getenv("EMAIL_RECEIVER", EMAIL_USER)
TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK_URL")
EMAIL_SUBJECT = os.getenv("EMAIL_SUBJECT", "🚨 EscalateAI Alert")

# Twilio (optional) for WhatsApp
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")  # e.g. whatsapp:+1415xxxxxxx

# DB and ID config
DB_PATH = os.getenv("ESCALE_DB_PATH", "escalations.db")
ESCALATION_PREFIX = "SESICE-25"

# NLP setup
analyzer = SentimentIntensityAnalyzer()
NEGATIVE_KEYWORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge","leak"],
    "dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "safety": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}

# In-memory processed hashes (for runtime dedupe), plus DB-backed dedupe (persistent)
processed_email_hashes = set()
processed_email_hashes_lock = threading.Lock()

# -----------------------
# Database / schema
# -----------------------
def ensure_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table with full schema if not exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS escalations (
        id TEXT PRIMARY KEY,
        customer TEXT,
        issue TEXT,
        sentiment TEXT,
        urgency TEXT,
        severity TEXT,
        criticality TEXT,
        category TEXT,
        status TEXT,
        timestamp TEXT,
        action_taken TEXT,
        owner TEXT,
        owner_email TEXT,
        escalated TEXT,
        priority TEXT,
        escalation_flag TEXT,
        action_owner TEXT,
        status_update_date TEXT,
        user_feedback TEXT
    )
    ''')

    """Check if a given hash already exists in the escalations table."""
    def hash_exists(h):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM escalations WHERE hash = ? LIMIT 1", (h,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists
    
    # Check if 'hash' column exists
    cursor.execute("PRAGMA table_info(escalations)")
    columns = [row[1] for row in cursor.fetchall()]
    hash_exists = 'hash' in columns

    if not hash_exists:
        cursor.execute("ALTER TABLE escalations ADD COLUMN hash TEXT")

    # Check if index exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='index' AND name='idx_escalations_hash'
    """)
    if not cursor.fetchone():
        cursor.execute("CREATE INDEX idx_escalations_hash ON escalations (hash)")

    conn.commit()
    conn.close()

def get_next_escalation_id():
    """Generate sequential escalation ID SESICE-25xxxxx by looking up max id in DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id FROM escalations WHERE id LIKE '{ESCALATION_PREFIX}%' ORDER BY id DESC LIMIT 1")
    last = cursor.fetchone()
    conn.close()
    if last:
        last_num_str = last[0].replace(ESCALATION_PREFIX, "")
        try:
            last_num = int(last_num_str)
        except Exception:
            last_num = 0
        next_num = last_num + 1
    else:
        next_num = 1
    return f"{ESCALATION_PREFIX}{str(next_num).zfill(5)}"

def insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag, priority="normal", owner="", owner_email="", h=None):
    """
    Insert escalation only if hash (h) not present. Returns tuple (inserted_bool, id_or_reason).
    If h is None, we still compute a hash from customer+issue for Excel paths.
    """
    if h is None:
        # fallback hash based on normalized customer + issue
        norm = f"{str(customer).strip()}|{str(issue).strip()}"
        h = hashlib.md5(norm.lower().encode('utf-8')).hexdigest()

    if hash_exists(h):
        return False, "duplicate"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    new_id = get_next_escalation_id()
    now = datetime.datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO escalations (
            id, customer, issue, sentiment, urgency, severity, criticality, category,
            status, timestamp, action_taken, owner, owner_email, escalated, priority, escalation_flag,
            action_owner, status_update_date, user_feedback, hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        new_id, customer, issue, sentiment, urgency, severity, criticality, category,
        "Open", now, "", owner, owner_email, escalation_flag, priority, escalation_flag,
        "", "", "", h
    ))
    conn.commit()
    conn.close()
    return True, new_id

def fetch_escalations_df():
    """Fetch entire escalations table as DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def update_escalation_status(esc_id, status=None, action_taken=None, action_owner=None, owner_email=None, feedback=None, sentiment=None, criticality=None):
    """Dynamic update to avoid overwriting nulls."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    updates = []
    params = []
    if status is not None:
        updates.append("status = ?"); params.append(status)
    if action_taken is not None:
        updates.append("action_taken = ?"); params.append(action_taken)
    if action_owner is not None:
        updates.append("action_owner = ?"); params.append(action_owner)
    if owner_email is not None:
        updates.append("owner_email = ?"); params.append(owner_email)
    if feedback is not None:
        updates.append("user_feedback = ?"); params.append(feedback)
    if sentiment is not None:
        updates.append("sentiment = ?"); params.append(sentiment)
    if criticality is not None:
        updates.append("criticality = ?"); params.append(criticality)
    updates.append("status_update_date = ?"); params.append(datetime.datetime.now().isoformat())
    params.append(esc_id)
    if updates:
        sql = f"UPDATE escalations SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(sql, params)
        conn.commit()
    conn.close()

# -----------------------
# Text cleaning, hashing, summarizing
# -----------------------
def normalize_text_for_hash(*parts):
    """Normalize string parts and return md5 hex digest."""
    joined = " | ".join([str(p) for p in parts if p is not None])
    cleaned = re.sub(r'\s+', ' ', joined).strip().lower()
    return hashlib.md5(cleaned.encode('utf-8')).hexdigest()

def generate_issue_hash_from_email(subject, body):
    """Normalized hash for emails to avoid duplicates including forwarded copies."""
    # Remove common forwarding metadata to be tolerant to forwards/headers
    patterns_to_strip = [
        r"[-]+[ ]*forwarded message[ ]*[-]+",
        r"from:.*", r"sent:.*", r"to:.*", r"subject:.*",
        r">.*", r"on .* wrote:"
    ]
    text = f"{subject or ''} {body or ''}"
    for p in patterns_to_strip:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def summarize_issue_text(issue_text, max_len=100):
    s = re.sub(r'\s+', ' ', str(issue_text)).strip()
    return s[:max_len] + "..." if len(s) > max_len else s

# -----------------------
# Email parsing with dedupe
# -----------------------
def parse_emails():
    """
    Connect to IMAP and fetch UNSEEN emails.
    For each email compute a normalized hash; if hash already in DB, skip.
    Returns list of parsed emails (customer, issue, raw, hash).
    """
    results = []
    try:
        IMAP = imaplib.IMAP4_SSL(EMAIL_SERVER)
        IMAP.login(EMAIL_USER, EMAIL_PASS)
        IMAP.select("inbox")
        status, messages = IMAP.search(None, "UNSEEN")
        if status != "OK":
            IMAP.logout()
            return results

        for num in messages[0].split():
            res, msg_data = IMAP.fetch(num, "(RFC822)")
            if res != "OK":
                continue
            for part in msg_data:
                if isinstance(part, tuple):
                    msg = email.message_from_bytes(part[1])
                    subj_raw = msg.get("Subject", "")
                    subj = decode_header(subj_raw)[0][0]
                    if isinstance(subj, bytes):
                        subj = subj.decode(errors='ignore')
                    from_ = msg.get("From", "unknown")
                    body = ""
                    if msg.is_multipart():
                        for p in msg.walk():
                            ctype = p.get_content_type()
                            disp = str(p.get("Content-Disposition"))
                            if ctype == "text/plain" and "attachment" not in disp:
                                try:
                                    body = p.get_payload(decode=True).decode(errors='ignore')
                                except Exception:
                                    body = str(p.get_payload())
                                break
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode(errors='ignore')
                        except Exception:
                            body = str(msg.get_payload())

                    h = generate_issue_hash_from_email(subj, body)

                    # Check persistent DB dedupe first
                    if hash_exists(h):
                        continue

                    # Also check runtime memory dedupe to avoid duplicates within same polling window
                    with processed_email_hashes_lock:
                        if h in processed_email_hashes:
                            continue
                        processed_email_hashes.add(h)

                    summary = summarize_issue_text(f"{subj} - {body}", max_len=120)
                    results.append({"customer": from_, "issue": summary, "raw": f"{subj}\n{body}", "hash": h})
        IMAP.logout()
    except Exception as e:
        st.error(f"Failed to parse emails: {e}")
    return results

# -----------------------
# NLP & tagging (VADER + keywords)
# -----------------------
def analyze_issue(issue_text):
    comp = analyzer.polarity_scores(issue_text)["compound"]
    if comp < -0.05:
        sentiment = "negative"
    elif comp > 0.05:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    text_l = str(issue_text).lower()
    urgency = "high" if any(k in text_l for kws in NEGATIVE_KEYWORDS.values() for k in kws) else "normal"

    category = None
    for cat, kws in NEGATIVE_KEYWORDS.items():
        if any(k in text_l for k in kws):
            category = cat
            break

    if category in ("safety", "technical"):
        severity = "critical"
    elif category in ("support", "business"):
        severity = "major"
    else:
        severity = "minor"

    criticality = "high" if (sentiment == "negative" and urgency == "high") else "medium"
    escalation_flag = "Yes" if urgency == "high" or sentiment == "negative" else "No"
    return sentiment, urgency, severity, criticality, category or "other", escalation_flag

# -----------------------
# ML model helpers
# -----------------------
def train_model():
    df = fetch_escalations_df()
    if df.shape[0] < 20:
        return None
    df = df.dropna(subset=['sentiment', 'urgency', 'severity', 'criticality', 'escalated'])
    if df.empty:
        return None
    X = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']].astype(str))
    y = df['escalated'].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    if y.nunique() < 2:
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    model._feature_names = X.columns.tolist()
    return model

def predict_escalation(model, sentiment, urgency, severity, criticality):
    if model is None:
        return "No"
    vec = {f"sentiment_{sentiment}": 1, f"urgency_{urgency}": 1, f"severity_{severity}": 1, f"criticality_{criticality}": 1}
    Xpred = pd.DataFrame([vec])
    Xpred = Xpred.reindex(columns=model._feature_names, fill_value=0)
    p = model.predict(Xpred)[0]
    return "Yes" if int(p) == 1 else "No"

# -----------------------
# Alerts: unified email + teams
# -----------------------
def send_alert(message, via="email", recipient=None):
    if via == "email":
        try:
            msg = EmailMessage()
            msg['Subject'] = EMAIL_SUBJECT
            msg['From'] = EMAIL_USER
            msg['To'] = recipient if recipient else ALERT_RECIPIENT
            msg.set_content(message)
            with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_USER, EMAIL_PASS)
                server.send_message(msg)
        except Exception as e:
            st.error(f"Email alert failed: {e}")
    elif via == "teams":
        try:
            if not TEAMS_WEBHOOK:
                st.error("MS Teams webhook not configured.")
                return
            r = requests.post(TEAMS_WEBHOOK, json={"text": message}, timeout=10)
            if r.status_code not in (200, 201):
                st.error(f"Teams alert failed: {r.status_code} {r.text}")
        except Exception as e:
            st.error(f"Teams alert failed: {e}")

# -----------------------
# WhatsApp via Twilio (optional)
# -----------------------
def send_whatsapp_message(phone_number, message):
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_FROM):
        # fallback to stub (UI will show a helpful message)
        return False, "Twilio not configured"
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        data = {
            "From": TWILIO_WHATSAPP_FROM,
            "To": f"whatsapp:{phone_number}",
            "Body": message
        }
        resp = requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=10)
        if resp.status_code in (200, 201):
            return True, "Sent"
        else:
            return False, f"Failed {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)

# -----------------------
# Background jobs
# -----------------------
def email_polling_job(poll_seconds=60):
    while True:
        try:
            emails = parse_emails()
            for e in emails:
                customer = e.get("customer", "unknown")
                issue = e.get("issue", "")
                h = e.get("hash")
                sentiment, urgency, severity, criticality, category, esc_flag = analyze_issue(issue)
                priority = "high" if severity == "critical" or esc_flag == "Yes" else "normal"
                inserted, info = insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, esc_flag, priority, "", "", h)
                # optionally log inserted or duplicate
        except Exception as ex:
            print("Email polling error:", ex)
        time.sleep(poll_seconds)

def send_daily_escalation_report():
    df = fetch_escalations_df()
    if df.empty:
        print("No escalations to report.")
        return
    df_esc = df[df["escalated"].str.lower() == "yes"] if "escalated" in df.columns else pd.DataFrame()
    if df_esc.empty:
        print("No escalated cases to send.")
        return
    file_path = "daily_escalated_cases.xlsx"
    try:
        df_esc.to_excel(file_path, index=False)
        msg = EmailMessage()
        msg['Subject'] = "📊 Daily Escalated Cases Report – 9 AM"
        msg['From'] = EMAIL_USER
        msg['To'] = ALERT_RECIPIENT
        msg.set_content("Attached is the daily escalation report as of 9 AM.")
        with open(file_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="daily_escalated_cases.xlsx")
        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as s:
            s.starttls()
            s.login(EMAIL_USER, EMAIL_PASS)
            s.send_message(msg)
        print("Daily escalation report sent.")
    except Exception as e:
        print("Failed to send daily report:", e)

def start_daily_scheduler():
    while True:
        now = datetime.datetime.now()
        if now.hour == 9 and now.minute == 0:
            try:
                send_daily_escalation_report()
            except Exception as e:
                print("Daily report error:", e)
            time.sleep(61)
        time.sleep(20)

# -----------------------
# UI: preserve labels, compact Kanban in columns
# -----------------------
ensure_schema()
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    /* Small UI tweaks to enable compact cards */
    .card {
        border-radius: 6px;
        padding: 8px;
        margin-bottom: 8px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }
    .card-leftbar {
        display:inline-block;
        width:6px;
        height:100%;
        vertical-align:top;
        margin-right:8px;
        border-radius:4px;
    }
    .compact-attr {
        font-size:12px;
        line-height:1.1;
        color:#333;
    }
    </style>
    <header>
        <div>
            <h1 style="margin: 0; padding-left: 20px;">🚨 EscalateAI - AI Based Escalation Prediction and Management Tool </h1>
        </div>
    </header>
    """,
    unsafe_allow_html=True
)

# Sidebar content preserved (exact names)
st.sidebar.markdown("""
    <style>
    .sidebar-title h2 { font-size:20px; margin-bottom:4px; font-weight:600; color:#2c3e50; text-align:center;}
    .sidebar-subtext { font-size:13px; color:#7f8c8d; text-align:center; margin-bottom:10px; }
    .sidebar-content { background-color:#ecf0f1; padding:10px; border-radius:8px; box-shadow:0px 2px 5px rgba(0,0,0,0.1); }
    </style>
    <div class="sidebar-title"><h2>⚙️ EscalateAI Controls</h2></div>
    <div class="sidebar-subtext">Manage, monitor & respond with agility.</div>
    <div class="sidebar-content"></div>
""", unsafe_allow_html=True)

# --- Excel upload (preserve labels & logic) ---
st.sidebar.header("📁 Upload Escalation Sheet")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])
if uploaded_file:
    try:
        df_excel = pd.read_excel(uploaded_file)
        st.sidebar.success("✅ Excel file loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to read Excel file: {e}")
        st.stop()
    required_columns = ["Customer", "Issue"]
    missing_cols = [col for col in required_columns if col not in df_excel.columns]
    if missing_cols:
        st.sidebar.error(f"Missing required columns: {', '.join(missing_cols)}")
        st.stop()
    if st.sidebar.button("🔍 Analyze & Insert"):
        processed_count = 0
        skipped = 0
        for idx, r in df_excel.iterrows():
            issue = str(r.get("Issue", "")).strip()
            customer = str(r.get("Customer", "Unknown")).strip()
            if not issue:
                st.warning(f"⚠️ Row {idx + 1} skipped: empty issue text.")
                skipped += 1
                continue
            # compute hash for Excel row -> normalized customer + issue
            h = normalize_text_for_hash(customer, issue)
            if hash_exists(h):
                skipped += 1
                continue
            issue_summary = summarize_issue_text(issue, max_len=120)
            sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
            inserted, info = insert_escalation(customer, issue_summary, sentiment, urgency, severity, criticality, category, escalation_flag, "normal", "", "", h)
            if inserted:
                processed_count += 1
        st.sidebar.success(f"🎯 {processed_count} rows processed successfully. {skipped} skipped as duplicates/empty.")

# --- Downloads (preserve labels) ---
st.sidebar.markdown("### 📤 Downloads")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("⬇️ All Complaints"):
        csv = fetch_escalations_df().to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="escalations.csv", mime="text/csv")
with col2:
    if st.button("⬇️ Escalated Only"):
        df_esc = fetch_escalations_df()
        df_esc = df_esc[df_esc["escalated"] == "Yes"] if not df_esc.empty else pd.DataFrame()
        if df_esc.empty:
            st.info("No escalated cases.")
        else:
            with pd.ExcelWriter("escalated_cases.xlsx") as writer:
                df_esc.to_excel(writer, index=False)
            with open("escalated_cases.xlsx", "rb") as f:
                st.download_button("Download Excel", f, file_name="escalated_cases.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- Email fetching (preserve label) ---
st.sidebar.markdown("### 📩 Email Integration")
if st.sidebar.button("Fetch Emails"):
    emails = parse_emails()
    for e in emails:
        issue = e["issue"]
        customer = e["customer"]
        h = e.get("hash")
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
        insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag, "normal", "", "", h)
    st.sidebar.success(f"✅ {len(emails)} emails processed")

# --- SLA Monitor (preserve label) ---
st.sidebar.markdown("### ⏰ SLA Monitor")
if st.sidebar.button("Trigger SLA Check"):
    df = fetch_escalations_df()
    if df.empty:
        st.sidebar.info("No records.")
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        breaches = df[(df['status'] != 'Resolved') & (df['priority'] == 'high') & ((datetime.datetime.now() - df['timestamp']) > datetime.timedelta(minutes=10))]
        if not breaches.empty:
            alert_msg = f"🚨 SLA breach for {len(breaches)} case(s)!"
            send_alert(alert_msg, via="teams")
            send_alert(alert_msg, via="email")
            st.sidebar.success("✅ Alerts sent")
        else:
            st.sidebar.info("All SLAs healthy")

# --- SLA banner (preserve behavior) ---
df_all = fetch_escalations_df()
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce') if not df_all.empty else pd.Series(dtype='datetime64[ns]')
breaches = df_all[(df_all['status'] != 'Resolved') & (df_all['priority'] == 'high') & ((datetime.datetime.now() - df_all['timestamp']) > datetime.timedelta(minutes=10))] if not df_all.empty else pd.DataFrame()
if not breaches.empty:
    st.sidebar.markdown(
        f"<div style='background:#dc3545;padding:8px;border-radius:5px;color:white;text-align:center;'><strong>🚨 {len(breaches)} SLA Breach(s) Detected</strong></div>",
        unsafe_allow_html=True
    )

# --- Filters (preserve labels) ---
st.sidebar.markdown("### 🔍 Escalation Filters")
df = fetch_escalations_df()
status_filter = st.sidebar.selectbox("Status", ["All", "Open", "In Progress", "Resolved"])
severity_filter = st.sidebar.selectbox("Severity", ["All"] + sorted(df["severity"].dropna().unique().tolist()) if not df.empty else ["All"])
sentiment_filter = st.sidebar.selectbox("Sentiment", ["All"] + sorted(df["sentiment"].dropna().unique().tolist()) if not df.empty else ["All"])
category_filter = st.sidebar.selectbox("Category", ["All"] + sorted(df["category"].dropna().unique().tolist()) if not df.empty else ["All"])
view = st.sidebar.radio("Escalation View", ["All", "Escalated", "Non-Escalated"])

filtered_df = df.copy() if not df.empty else pd.DataFrame()
if status_filter != "All" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["status"] == status_filter]
if severity_filter != "All" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["severity"] == severity_filter]
if sentiment_filter != "All" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["sentiment"] == sentiment_filter]
if category_filter != "All" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["category"] == category_filter]
if view == "Escalated" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["escalated"] == "Yes"]
elif view == "Non-Escalated" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["escalated"] != "Yes"]

# --- Manual alerts (preserve labels) ---
st.sidebar.markdown("### 🔔 Manual Notifications")
manual_msg = st.sidebar.text_area("Compose Alert", "🚨 Test alert from EscalateAI")
if st.sidebar.button("Send MS Teams"):
    send_alert(manual_msg, via="teams")
    st.sidebar.success("✅ MS Teams alert sent")
if st.sidebar.button("Send Email"):
    send_alert(manual_msg, via="email")
    st.sidebar.success("✅ Email alert sent")

# --- WhatsApp Alerts (preserve labels) ---
st.sidebar.markdown("### 📲 WhatsApp Alerts")
status_check = st.sidebar.selectbox("Case Status", ["Open", "In Progress", "Resolved"])
if status_check == "Resolved":
    df_resolved = fetch_escalations_df()
    df_resolved = df_resolved[df_resolved["status"] == "Resolved"] if not df_resolved.empty else pd.DataFrame()
    if not df_resolved.empty:
        escalation_id = st.sidebar.selectbox("🔢 Select Resolved Escalation ID", df_resolved["id"].tolist())
        phone = st.sidebar.text_input("📞 Phone Number", "+91", help="Include country code (e.g., +91)")
        msg_wa = st.sidebar.text_area("📨 Message", f"Your issue with ID {escalation_id} has been resolved. Thank you!")
        if st.sidebar.button("Send WhatsApp"):
            ok, info = send_whatsapp_message(phone, msg_wa)
            if ok:
                st.sidebar.success(f"✅ WhatsApp sent to {phone} for Escalation ID {escalation_id}")
            else:
                st.sidebar.warning(f"⚠️ WhatsApp not sent: {info} (Set TWILIO env vars to enable)")
    else:
        st.sidebar.warning("No resolved escalations found.")
else:
    st.sidebar.info("WhatsApp alerts are only available for 'Resolved' cases.")

# --- Downloads / Dev options preserved ---
if st.sidebar.checkbox("🧪 View Raw Database"):
    st.sidebar.dataframe(fetch_escalations_df())

if st.sidebar.button("🗑️ Reset Database (Dev Only)"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS escalations")
    conn.commit()
    conn.close()
    ensure_schema()
    st.sidebar.warning("Database reset. Please restart the app.")

if st.sidebar.button("📤 Send Daily Report Now"):
    send_daily_escalation_report()

# --- Tabs (preserve tab names) ---
tabs = st.tabs(["🗃️ All", "🚩 Escalated", "🔁 Feedback & Retraining"])

# Color mapping for statuses
STATUS_HEADER_COLORS = {
    "Open": "#e74c3c",         # Red
    "In Progress": "#f39c12",  # Orange
    "Resolved": "#2ecc71"      # Green
}
# Left border color for cards by status (same mapping)
CARD_LEFTBAR_COLORS = STATUS_HEADER_COLORS.copy()

# Compact Kanban board (columns)
with tabs[0]:
    st.subheader("📊 Escalation Kanban Board")
    df = filtered_df.copy() if not filtered_df.empty else pd.DataFrame()
    counts = df['status'].value_counts() if not df.empty else pd.Series(dtype=int)
    open_count = counts.get('Open', 0)
    inprogress_count = counts.get('In Progress', 0)
    resolved_count = counts.get('Resolved', 0)
    st.markdown(f"**Open:** {open_count} | **In Progress:** {inprogress_count} | **Resolved:** {resolved_count}")

    col1, col2, col3 = st.columns(3)
    for status_name, col in zip(["Open", "In Progress", "Resolved"], [col1, col2, col3]):
        with col:
            header_color = STATUS_HEADER_COLORS.get(status_name, "#FFA500")
            col.markdown(f"<h3 style='background-color:{header_color};color:white;padding:8px;border-radius:5px;text-align:center;'>{status_name}</h3>", unsafe_allow_html=True)
            bucket = df[df["status"] == status_name] if not df.empty else pd.DataFrame()
            # Show as compact cards with left color bar
            for _, r in bucket.iterrows():
                esc_id = r.get("id", "")
                left_color = CARD_LEFTBAR_COLORS.get(status_name, "#888")
                flag = "🚩" if str(r.get("escalated","")).strip().lower() == "yes" else ""
                title = f"{esc_id} - {r.get('customer','')} {flag}"
                # compact card HTML
                card_html = f"""
                <div class="card">
                    <div style="display:flex;align-items:flex-start;">
                        <div class="card-leftbar" style="background:{left_color};"></div>
                        <div style="flex:1;">
                            <div style="font-weight:600;font-size:14px;margin-bottom:4px;">{title}</div>
                            <div class="compact-attr">
                                Severity: {r.get('severity','minor')} &nbsp;|&nbsp;
                                Criticality: {r.get('criticality','medium')} &nbsp;|&nbsp;
                                Category: {r.get('category','other')}
                                <br>
                                Sentiment: {r.get('sentiment','positive')} &nbsp;|&nbsp;
                                Urgency: {r.get('urgency','normal')} &nbsp;|&nbsp;
                                Escalated: {r.get('escalated','No')}
                            </div>
                        </div>
                    </div>
                </div>
                """
                # expander with very compact content preserved
                with st.expander("", expanded=False):
                    # Show same compact header then actions
                    st.markdown(card_html, unsafe_allow_html=True)
                    # Ageing
                    try:
                        ts = pd.to_datetime(r.get("timestamp"))
                        delta = datetime.datetime.now() - ts.to_pydatetime()
                        days = delta.days
                        hours = delta.seconds // 3600
                        mins = (delta.seconds % 3600) // 60
                        age_str = f"{days}d {hours}h {mins}m"
                        total_hours = delta.total_seconds() / 3600
                        if total_hours < 4:
                            age_color = "#2ecc71"
                        elif total_hours < 12:
                            age_color = "#f1c40f"
                        else:
                            age_color = "#e74c3c"
                    except Exception:
                        age_str = "N/A"; age_color = "#7f8c8d"
                    st.markdown(f"**⏱️ Ageing:** <span style='color:{age_color}; font-weight:bold'>{age_str}</span>", unsafe_allow_html=True)

                    # Actions (preserve button labels)
                    c1, c2, c3 = st.columns([1,1,2])
                    if c1.button("✔️ Mark as Resolved", key=f"resolve_{esc_id}"):
                        owner_email = r.get("owner_email", EMAIL_USER)
                        update_escalation_status(esc_id, status="Resolved", owner_email=owner_email)
                        send_alert("Case marked as resolved.", via="email", recipient=owner_email)
                        send_alert("Case marked as resolved.", via="teams")
                        st.experimental_rerun()

                    n1_email = c2.text_input("N+1 Email", key=f"n1email_{esc_id}")
                    if c3.button("🚀 Escalate to N+1", key=f"escalate_{esc_id}"):
                        update_escalation_status(esc_id, status=r.get("status","Open"), owner_email=n1_email)
                        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
                        cur.execute("UPDATE escalations SET escalated = 'Yes' WHERE id = ?", (esc_id,))
                        conn.commit(); conn.close()
                        send_alert("Case escalated to N+1.", via="email", recipient=n1_email)
                        send_alert("Case escalated to N+1.", via="teams")
                        st.experimental_rerun()

                    # Editable fields (compact)
                    new_status = st.selectbox("Update Status", ["Open","In Progress","Resolved"], index=["Open","In Progress","Resolved"].index(r.get("status","Open")), key=f"status_{esc_id}")
                    new_action = st.text_input("Action Taken", r.get("action_taken",""), key=f"action_{esc_id}")
                    new_owner = st.text_input("Owner", r.get("owner",""), key=f"owner_{esc_id}")
                    new_owner_email = st.text_input("Owner Email", r.get("owner_email",""), key=f"email_{esc_id}")

                    if st.button("💾 Save Changes", key=f"save_{esc_id}"):
                        update_escalation_status(esc_id, status=new_status, action_taken=new_action, action_owner=new_action, owner_email=new_owner_email)
                        note = f"Case {esc_id} updated. Status: {new_status}. Action: {new_action}."
                        send_alert(note, via="email", recipient=new_owner_email)
                        send_alert(note, via="teams")
                        st.success("Saved.")
                        st.experimental_rerun()

with tabs[1]:
    st.subheader("🚩 Escalated Issues")
    df_all = fetch_escalations_df()
    df_esc = df_all[df_all["escalated"] == "Yes"] if not df_all.empty else pd.DataFrame()
    st.dataframe(df_esc)

with tabs[2]:
    st.subheader("🔁 Feedback & Retraining")
    df_fb = fetch_escalations_df()
    df_feedback = df_fb[df_fb["escalated"].notnull()] if not df_fb.empty else pd.DataFrame()
    for _, row in (df_feedback.head(200).iterrows() if not df_feedback.empty else []):
        with st.expander(f"🆔 {row['id']}"):
            fb = st.selectbox("Escalation Accuracy", ["Correct", "Incorrect"], key=f"fb_{row['id']}")
            sent = st.selectbox("Sentiment", ["Positive", "Neutral", "Negative"], key=f"sent_{row['id']}")
            crit = st.selectbox("Criticality", ["Low", "Medium", "High", "Urgent"], key=f"crit_{row['id']}")
            notes = st.text_area("Notes", key=f"note_{row['id']}")
            if st.button("Submit", key=f"btn_{row['id']}"):
                owner_email = row.get("owner_email", EMAIL_USER)
                update_escalation_status(row['id'], status="Resolved", action_taken=row.get("action_taken",""), action_owner=row.get("owner",""), owner_email=owner_email)
                send_alert("Case marked as resolved.", via="email", recipient=owner_email)
                st.success("Feedback saved.")
    if st.button("🔁 Retrain Model"):
        st.info("Retraining model with feedback (may take a few seconds)...")
        model = train_model()
        if model:
            st.success("Model retrained successfully.")
        else:
            st.warning("Not enough data to retrain model.")

# Background threads (only once)
if 'email_thread' not in st.session_state:
    email_thread = threading.Thread(target=email_polling_job, daemon=True)
    email_thread.start()
    st.session_state['email_thread'] = email_thread

if 'daily_scheduler' not in st.session_state:
    scheduler_thread = threading.Thread(target=start_daily_scheduler, daemon=True)
    scheduler_thread.start()
    st.session_state['daily_scheduler'] = scheduler_thread

# Dev options preserved at bottom of sidebar earlier (kept)

st.markdown("---")
st.markdown("**Notes & next steps:**")
st.markdown("""
- Ensure `.env` contains: EMAIL_USER, EMAIL_PASS, EMAIL_SERVER, EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, EMAIL_RECEIVER, MS_TEAMS_WEBHOOK_URL (optional).
- For WhatsApp sending: set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM (format 'whatsapp:+12345...').
- Duplicate detection is MD5 on normalized content; for emails it's subject+body normalized (forwards/headers stripped). For Excel it's normalized Customer+Issue.
- If you want dedupe to persist across app copies, the DB hash column already stores hashes (good).
- Run with `streamlit run EscalateAIV610082025.py`.
""")
