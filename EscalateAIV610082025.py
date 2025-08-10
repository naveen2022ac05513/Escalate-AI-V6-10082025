# escalate_ai.py
# Corrected, sequenced, and preserved-UI version of your uploaded app
# Adds optional WhatsApp via Twilio while keeping all UI names exactly as uploaded.

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
# Load environment once
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

# Twilio for WhatsApp (optional)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")  # e.g., "whatsapp:+1415XXXXXXX"

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

# In-memory dedupe for emails (optional persistent storage can be added)
processed_email_hashes = set()
processed_email_hashes_lock = threading.Lock()

# -----------------------
# Database helpers
# -----------------------
def ensure_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
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
    conn.commit()
    conn.close()

def get_next_escalation_id():
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

def insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag, priority="normal", owner="", owner_email=""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    new_id = get_next_escalation_id()
    now = datetime.datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO escalations (
            id, customer, issue, sentiment, urgency, severity, criticality, category,
            status, timestamp, action_taken, owner, owner_email, escalated, priority, escalation_flag,
            action_owner, status_update_date, user_feedback
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        new_id, customer, issue, sentiment, urgency, severity, criticality, category,
        "Open", now, "", owner, owner_email, escalation_flag, priority, escalation_flag,
        "", "", ""
    ))
    conn.commit()
    conn.close()
    return new_id

def fetch_escalations():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def update_escalation_status(esc_id, status=None, action_taken=None, action_owner=None, owner_email=None, feedback=None, sentiment=None, criticality=None):
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
    sql = f"UPDATE escalations SET {', '.join(updates)} WHERE id = ?"
    cursor.execute(sql, params)
    conn.commit()
    conn.close()

# -----------------------
# Text helpers & dedupe
# -----------------------
def generate_issue_hash(issue_text):
    patterns_to_remove = [
        r"[-]+[ ]*Forwarded message[ ]*[-]+",
        r"From:.*", r"Sent:.*", r"To:.*", r"Subject:.*",
        r">.*",
        r"On .* wrote:"
    ]
    txt = issue_text
    for p in patterns_to_remove:
        txt = re.sub(p, " ", txt, flags=re.IGNORECASE)
    txt = re.sub(r'\s+', ' ', txt).strip().lower()
    return hashlib.md5(txt.encode('utf-8')).hexdigest()

def summarize_issue_text(issue_text):
    clean_text = re.sub(r'\s+', ' ', str(issue_text)).strip()
    return clean_text[:120] + "..." if len(clean_text) > 120 else clean_text

# -----------------------
# Email parsing (IMAP)
# -----------------------
def parse_emails():
    emails_out = []
    try:
        imap = imaplib.IMAP4_SSL(EMAIL_SERVER)
        imap.login(EMAIL_USER, EMAIL_PASS)
        imap.select("inbox")
        status, messages = imap.search(None, "UNSEEN")
        if status != "OK":
            imap.logout()
            return emails_out
        for num in messages[0].split():
            res, msg_data = imap.fetch(num, "(RFC822)")
            if res != "OK":
                continue
            for part in msg_data:
                if isinstance(part, tuple):
                    msg = email.message_from_bytes(part[1])
                    subject = decode_header(msg.get("Subject", ""))[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(errors='ignore')
                    from_ = msg.get("From", "unknown")
                    body = ""
                    if msg.is_multipart():
                        for p in msg.walk():
                            if p.get_content_type() == "text/plain" and "attachment" not in str(p.get("Content-Disposition")):
                                try:
                                    body = p.get_payload(decode=True).decode(errors='ignore')
                                except:
                                    body = str(p.get_payload())
                                break
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode(errors='ignore')
                        except:
                            body = str(msg.get_payload())
                    full_text = f"{subject} - {body}"
                    h = generate_issue_hash(full_text)
                    with processed_email_hashes_lock:
                        if h in processed_email_hashes:
                            continue
                        processed_email_hashes.add(h)
                    summary = summarize_issue_text(full_text)
                    emails_out.append({"customer": from_, "issue": summary, "raw": full_text})
        imap.logout()
    except Exception as e:
        st.error(f"Failed to parse emails: {e}")
    return emails_out

# -----------------------
# NLP & Tagging
# -----------------------
def analyze_issue(issue_text):
    compound = analyzer.polarity_scores(issue_text)["compound"]
    if compound < -0.05:
        sentiment = "negative"
    elif compound > 0.05:
        sentiment = "positive"
    else:
        sentiment = "neutral"
    text_l = issue_text.lower()
    urgency = "high" if any(word in text_l for cat in NEGATIVE_KEYWORDS.values() for word in cat) else "normal"
    category = None
    for cat, kws in NEGATIVE_KEYWORDS.items():
        if any(k in text_l for k in kws):
            category = cat
            break
    if category in ["safety", "technical"]:
        severity = "critical"
    elif category in ["support", "business"]:
        severity = "major"
    else:
        severity = "minor"
    criticality = "high" if sentiment == "negative" and urgency == "high" else "medium"
    escalation_flag = "Yes" if urgency == "high" or sentiment == "negative" else "No"
    return sentiment, urgency, severity, criticality, category or "other", escalation_flag

# -----------------------
# ML model
# -----------------------
def train_model():
    df = fetch_escalations()
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
    vec = {
        f"sentiment_{sentiment}": 1,
        f"urgency_{urgency}": 1,
        f"severity_{severity}": 1,
        f"criticality_{criticality}": 1
    }
    Xpred = pd.DataFrame([vec])
    Xpred = Xpred.reindex(columns=model._feature_names, fill_value=0)
    pred = model.predict(Xpred)[0]
    return "Yes" if int(pred) == 1 else "No"

# -----------------------
# Alerts (Email + Teams)
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
            response = requests.post(TEAMS_WEBHOOK, json={"text": message})
            if response.status_code not in (200,201):
                st.error(f"Teams alert failed: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Teams alert failed: {e}")

# -----------------------
# WhatsApp via Twilio (optional)
# -----------------------
def send_whatsapp_message(phone_number, message):
    """
    Send WhatsApp message via Twilio if TWILIO credentials present.
    phone_number should include country code, e.g., +91XXXXXXXXXX
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_FROM):
        # fallback to stub behaviour (UI expected to show success but we note stub)
        print("Twilio not configured — WhatsApp not sent. Provide TWILIO_* env vars to enable.")
        return False, "Twilio credentials not configured. Message not sent."
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
            return False, f"Failed: {resp.status_code} {resp.text}"
    except Exception as e:
        return False, str(e)

# -----------------------
# Background: email poll + daily report
# -----------------------
def email_polling_job():
    while True:
        try:
            emails = parse_emails()
            for e in emails:
                issue = e["issue"]
                customer = e["customer"]
                sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
                priority = "high" if severity == "critical" or escalation_flag == "Yes" else "normal"
                insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag, priority)
        except Exception as ex:
            print("Email polling error:", ex)
        time.sleep(60)

def send_daily_escalation_report():
    df = fetch_escalations()
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
# UI: Sequence preserved and names unchanged from uploaded file
# -----------------------
ensure_schema()
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* Keep your original styling space (minimal) */
    </style>
    <header>
        <div>
            <h1 style="margin: 0; padding-left: 20px;">🚨 EscalateAI - AI Based Escalation Prediction and Management Tool </h1>
        </div>
    </header>
    """,
    unsafe_allow_html=True
)

# The original file rendered a status bar using a 'row' variable erroneously.
# We will render such status bars inside expanders only (preserving UI experience).

# Sidebar styling and controls (preserve exact names)
st.sidebar.markdown("""
    <style>
    .sidebar-title h2 {
        font-size: 20px;
        margin-bottom: 4px;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
    }
    .sidebar-subtext {
        font-size: 13px;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 10px;
    }
    .sidebar-content {
        background-color: #ecf0f1;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    <div class="sidebar-title">
        <h2>⚙️ EscalateAI Controls</h2>
    </div>
    <div class="sidebar-subtext">
        Manage, monitor & respond with agility.
    </div>
    <div class="sidebar-content">
        <!-- Sidebar content preserved -->
    </div>
""", unsafe_allow_html=True)

# --- Upload Excel (exact labels preserved) ---
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
        for idx, row in df_excel.iterrows():
            issue = str(row.get("Issue", "")).strip()
            customer = str(row.get("Customer", "Unknown")).strip()
            if not issue:
                st.warning(f"⚠️ Row {idx + 1} skipped: empty issue text.")
                continue
            issue_summary = summarize_issue_text(issue)
            sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
            insert_escalation(customer, issue_summary, sentiment, urgency, severity, criticality, category, escalation_flag)
            processed_count += 1
        st.sidebar.success(f"🎯 {processed_count} rows processed successfully.")

# --- Downloads (preserve labels) ---
st.sidebar.markdown("### 📤 Downloads")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("⬇️ All Complaints"):
        csv = fetch_escalations().to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="escalations.csv", mime="text/csv")
with col2:
    if st.button("⬇️ Escalated Only"):
        df_esc = fetch_escalations()
        df_esc = df_esc[df_esc["escalated"] == "Yes"] if not df_esc.empty else pd.DataFrame()
        if df_esc.empty:
            st.info("No escalated cases.")
        else:
            with pd.ExcelWriter("escalated_cases.xlsx") as writer:
                df_esc.to_excel(writer, index=False)
            with open("escalated_cases.xlsx", "rb") as f:
                st.download_button("Download Excel", f, file_name="escalated_cases.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- Email Fetching (preserve button name) ---
st.sidebar.markdown("### 📩 Email Integration")
if st.sidebar.button("Fetch Emails"):
    emails = parse_emails()
    for e in emails:
        issue, customer = e["issue"], e["customer"]
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
        insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
    st.sidebar.success(f"✅ {len(emails)} emails processed")

# --- SLA Monitor (preserve name) ---
st.sidebar.markdown("### ⏰ SLA Monitor")
if st.sidebar.button("Trigger SLA Check"):
    df = fetch_escalations()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') if not df.empty else pd.Series(dtype='datetime64[ns]')
    breaches = df[(df['status'] != 'Resolved') & (df['priority'] == 'high') &
                  ((datetime.datetime.now() - df['timestamp']) > datetime.timedelta(minutes=10))] if not df.empty else pd.DataFrame()
    if not breaches.empty:
        alert_msg = f"🚨 SLA breach for {len(breaches)} case(s)!"
        send_alert(alert_msg, via="teams")
        send_alert(alert_msg, via="email")
        st.sidebar.success("✅ Alerts sent")
    else:
        st.sidebar.info("All SLAs healthy")

# --- SLA Banner (preserve behavior) ---
df_all = fetch_escalations()
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce') if not df_all.empty else pd.Series(dtype='datetime64[ns]')
breaches = df_all[(df_all['status'] != 'Resolved') & (df_all['priority'] == 'high') &
                  ((datetime.datetime.now() - df_all['timestamp']) > datetime.timedelta(minutes=10))] if not df_all.empty else pd.DataFrame()
if not breaches.empty:
    st.sidebar.markdown(
        f"<div style='background:#dc3545;padding:8px;border-radius:5px;color:white;text-align:center;'>"
        f"<strong>🚨 {len(breaches)} SLA Breach(s) Detected</strong></div>",
        unsafe_allow_html=True
    )

# --- Filters (preserve labels) ---
st.sidebar.markdown("### 🔍 Escalation Filters")
df = fetch_escalations()
status = st.sidebar.selectbox("Status", ["All", "Open", "In Progress", "Resolved"])
severity = st.sidebar.selectbox("Severity", ["All"] + sorted(df["severity"].dropna().unique().tolist()) if not df.empty else ["All"])
sentiment = st.sidebar.selectbox("Sentiment", ["All"] + sorted(df["sentiment"].dropna().unique().tolist()) if not df.empty else ["All"])
category = st.sidebar.selectbox("Category", ["All"] + sorted(df["category"].dropna().unique().tolist()) if not df.empty else ["All"])
view = st.sidebar.radio("Escalation View", ["All", "Escalated", "Non-Escalated"])

filtered_df = df.copy() if not df.empty else pd.DataFrame()
if status != "All" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["status"] == status]
if severity != "All" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["severity"] == severity]
if sentiment != "All" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["sentiment"] == sentiment]
if category != "All" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["category"] == category]
if view == "Escalated" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["escalated"] == "Yes"]
elif view == "Non-Escalated" and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df["escalated"] != "Yes"]

# --- Manual Alerts (preserve labels) ---
st.sidebar.markdown("### 🔔 Manual Notifications")
msg = st.sidebar.text_area("Compose Alert", "🚨 Test alert from EscalateAI")
if st.sidebar.button("Send MS Teams"):
    send_alert(msg, via="teams")
    st.sidebar.success("✅ MS Teams alert sent")
if st.sidebar.button("Send Email"):
    send_alert(msg, via="email")
    st.sidebar.success("✅ Email alert sent")

# --- WhatsApp Alerts (preserve labels exactly) ---
st.sidebar.markdown("### 📲 WhatsApp Alerts")
status_check = st.sidebar.selectbox("Case Status", ["Open", "In Progress", "Resolved"])

if status_check == "Resolved":
    df_resolved = fetch_escalations()
    df_resolved = df_resolved[df_resolved["status"] == "Resolved"] if not df_resolved.empty else pd.DataFrame()
    if not df_resolved.empty:
        escalation_id = st.sidebar.selectbox(
            "🔢 Select Resolved Escalation ID",
            df_resolved["id"].tolist()
        )
        phone = st.sidebar.text_input("📞 Phone Number", "+91", help="Include country code (e.g., +91)")
        msg_wa = st.sidebar.text_area("📨 Message", f"Your issue with ID {escalation_id} has been resolved. Thank you!")
        if st.sidebar.button("Send WhatsApp"):
            success, info = send_whatsapp_message(phone, msg_wa)
            if success:
                st.sidebar.success(f"✅ WhatsApp sent to {phone} for Escalation ID {escalation_id}")
            else:
                st.sidebar.warning(f"⚠️ WhatsApp not sent: {info}\n(If you want real WhatsApp sending, set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM in .env)")
    else:
        st.sidebar.warning("No resolved escalations found.")
else:
    st.sidebar.info("WhatsApp alerts are only available for 'Resolved' cases.")

# --- Secondary email / teams alert function redefinition removed (kept single send_alert) ---

# --- Tabs (preserve tab names) ---
tabs = st.tabs(["🗃️ All", "🚩 Escalated", "🔁 Feedback & Retraining"])

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
            col.markdown(f"<h3 style='background-color:#FFA500;color:white;padding:8px;border-radius:5px;text-align:center;'>{status_name}</h3>", unsafe_allow_html=True)
            bucket = df[df["status"] == status_name] if not df.empty else pd.DataFrame()
            for i, row in bucket.iterrows():
                flag = "🚩" if row.get('escalated','') == 'Yes' else ""
                summary = summarize_issue_text(row.get('issue',''))
                expander_label = f"{row.get('id','')} - {row.get('customer','')} {flag} – {summary}"
                with st.expander(expander_label, expanded=False):
                    colA, colB, colC = st.columns(3)
                    try:
                        timestamp = pd.to_datetime(row.get("timestamp"))
                        now = datetime.datetime.now()
                        ageing_timedelta = now - timestamp.to_pydatetime()
                        days = ageing_timedelta.days
                        hours, remainder = divmod(ageing_timedelta.seconds, 3600)
                        minutes, _ = divmod(remainder, 60)
                        ageing_str = f"{days}d {hours}h {minutes}m"
                        total_hours = ageing_timedelta.total_seconds() / 3600
                        if total_hours < 12:
                            ageing_color = "#2ecc71"
                        elif 12 <= total_hours < 24:
                            ageing_color = "#e67e22"
                        else:
                            ageing_color = "#e74c3c"
                    except Exception:
                        ageing_str = "N/A"
                        ageing_color = "#7f8c8d"

                    st.markdown(f"**⏱️ Ageing:** <span style='color:{ageing_color}; font-weight:bold;'>{ageing_str}</span>", unsafe_allow_html=True)

                    if colA.button("✔️ Mark as Resolved", key=f"resolved_{row.get('id','')}"):
                        owner_email = row.get("owner_email", EMAIL_USER)
                        update_escalation_status(row.get('id'), status="Resolved", owner_email=owner_email)
                        send_alert("Case marked as resolved.", via="email", recipient=owner_email)
                        send_alert("Case marked as resolved.", via="teams")
                        st.experimental_rerun()

                    n1_email = colB.text_input("N+1 Email", key=f"n1email_{row.get('id','')}")
                    if colC.button("🚀 Escalate to N+1", key=f"n1btn_{row.get('id','')}"):
                        update_escalation_status(row.get('id'), status=row.get("status","Open"), owner_email=n1_email)
                        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
                        cur.execute("UPDATE escalations SET escalated = 'Yes' WHERE id = ?", (row.get('id'),))
                        conn.commit(); conn.close()
                        send_alert("Case escalated to N+1.", via="email", recipient=n1_email)
                        send_alert("Case escalated to N+1.", via="teams")
                        st.experimental_rerun()

                    st.markdown(f"**Issue:** {row.get('issue','')}")
                    st.markdown(f"**Severity:** {row.get('severity','')}")
                    st.markdown(f"**Criticality:** {row.get('criticality','')}")
                    st.markdown(f"**Category:** {row.get('category','')}")
                    st.markdown(f"**Sentiment:** {row.get('sentiment','')}")
                    st.markdown(f"**Urgency:** {row.get('urgency','')}")
                    st.markdown(f"**Escalated:** {row.get('escalated','') or 'No'}")

                    new_status = st.selectbox("Update Status", ["Open", "In Progress", "Resolved"],
                                              index=["Open", "In Progress", "Resolved"].index(row.get("status","Open")),
                                              key=f"status_{row.get('id','')}")
                    new_action = st.text_input("Action Taken", row.get("action_taken",""), key=f"action_{row.get('id','')}")
                    new_owner = st.text_input("Owner", row.get("owner",""), key=f"owner_{row.get('id','')}")
                    new_owner_email = st.text_input("Owner Email", row.get("owner_email",""), key=f"email_{row.get('id','')}")

                    if st.button("💾 Save Changes", key=f"save_{row.get('id','')}"):
                        update_escalation_status(row.get('id'), status=new_status, action_taken=new_action, action_owner=new_action, owner_email=new_owner_email)
                        st.success("Escalation updated.")
                        notification_message = f"""
                        🔔 Hello {new_owner},

                        The escalation case #{row.get('id')} assigned to you has been updated:

                        • Status: {new_status}
                        • Action Taken: {new_action}
                        • Category: {row.get('category')}
                        • Severity: {row.get('severity')}
                        • Urgency: {row.get('urgency')}
                        • Sentiment: {row.get('sentiment')}

                        Please review the updates on the EscalateAI dashboard.
                        """
                        send_alert(notification_message.strip(), via="email", recipient=new_owner_email)
                        send_alert(notification_message.strip(), via="teams")
                        st.experimental_rerun()

with tabs[1]:
    st.subheader("🚩 Escalated Issues")
    df = filtered_df.copy() if not filtered_df.empty else pd.DataFrame()
    df_esc = df[df["escalated"] == "Yes"] if not df.empty else pd.DataFrame()
    st.dataframe(df_esc)

with tabs[2]:
    st.subheader("🔁 Feedback & Retraining")
    df_fb = fetch_escalations()
    df_feedback = df_fb[df_fb["escalated"].notnull()] if not df_fb.empty else pd.DataFrame()
    fb_map = {"Correct": 1, "Incorrect": 0}
    for _, row in (df_feedback.head(100).iterrows() if not df_feedback.empty else []):
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

# Background threads start (only once)
if 'email_thread' not in st.session_state:
    email_thread = threading.Thread(target=email_polling_job, daemon=True)
    email_thread.start()
    st.session_state['email_thread'] = email_thread

if 'daily_scheduler' not in st.session_state:
    scheduler_thread = threading.Thread(target=start_daily_scheduler, daemon=True)
    scheduler_thread.start()
    st.session_state['daily_scheduler'] = scheduler_thread

# Dev options preserved
if st.sidebar.checkbox("🧪 View Raw Database"):
    df = fetch_escalations()
    st.sidebar.dataframe(df)

if st.sidebar.button("🗑️ Reset Database (Dev Only)"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS escalations")
    conn.commit()
    conn.close()
    st.sidebar.warning("Database reset. Please restart the app.")

# Daily report send (kept, also accessible via sidebar)
if st.sidebar.button("📤 Send Daily Report Now"):
    send_daily_escalation_report()

# End notes preserved
st.markdown("---")
st.markdown("**Notes & next steps:**")
st.markdown("""
- Ensure your .env contains the required credentials: EMAIL_USER, EMAIL_PASS, EMAIL_SERVER, EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, EMAIL_RECEIVER, MS_TEAMS_WEBHOOK_URL (optional).
- For WhatsApp sending, set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM (format 'whatsapp:+1234...').
- For persistent email dedupe across restarts, add a DB table for processed hashes.
- Run with: `streamlit run escalate_ai.py`
""")
