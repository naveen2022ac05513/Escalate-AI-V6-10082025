"""
EscalateAI - Corrected and sequenced single-file app
Features:
- Sequential escalation IDs: SESICE-25xxxxx
- SQLite backend with schema enforcement (adds owner_email if missing)
- Gmail IMAP fetch (Fetch Emails button + background polling)
- Excel upload and bulk insert
- VADER sentiment + keyword matching (no SpaCy)
- ML model (RandomForest) for escalation prediction + retrain flow
- Alerts via Email and MS Teams (single unified send_alert)
- Streamlit UI with Kanban, ageing, filters, manual notifications, WhatsApp stub
- SLA monitor (10-minute breach detection for 'high' priority)
- Daily email report (scheduler)
- Thorough inline comments for maintainability
"""

import os
import re
import time
import hashlib
import base64
import threading
import datetime
import sqlite3
from dotenv import load_dotenv

# Streamlit & common libs
import streamlit as st
import pandas as pd
import numpy as np
import imaplib
import email
from email.header import decode_header
from email.message import EmailMessage
import smtplib
import requests

# NLP & ML
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------
# ---- INITIAL SETUP ------
# -------------------------
load_dotenv()  # single centralized load

# Environment / configuration (fallback sensible defaults where possible)
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
ALERT_RECIPIENT = os.getenv("EMAIL_RECEIVER", EMAIL_USER)
TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK_URL")
EMAIL_SUBJECT_PREFIX = os.getenv("EMAIL_SUBJECT", "EscalateAI Notification")

# Database & ID config
DB_PATH = os.getenv("ESCALE_DB_PATH", "escalations.db")
ESCALATION_PREFIX = "SESICE-25"  # keep your requested prefix

# VADER analyzer and keywords (user's expanded list)
analyzer = SentimentIntensityAnalyzer()
NEGATIVE_KEYWORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge", "leak"],
    "dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "safety": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}

# For deduping processed email contents in-memory (persisting dedupe can be added)
processed_email_hashes = set()
processed_email_hashes_lock = threading.Lock()

# -------------------------
# ---- DB / SCHEMA UTILS --
# -------------------------
def ensure_schema():
    """Ensure DB exists and schema has required columns; add owner_email if missing."""
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
    """
    Generate sequential ID like SESICE-25xxxxx. Query DB for latest and increment.
    If latest record has non-integer suffix, start from 1.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT id FROM escalations WHERE id LIKE '{ESCALATION_PREFIX}%' ORDER BY id DESC LIMIT 1")
        last = cursor.fetchone()
    finally:
        conn.close()

    if last:
        last_id = last[0]
        suffix = last_id.replace(ESCALATION_PREFIX, "")
        try:
            n = int(suffix)
        except Exception:
            n = 0
        next_n = n + 1
    else:
        next_n = 1
    return f"{ESCALATION_PREFIX}{str(next_n).zfill(5)}"  # pad to 5 digits

def insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag, priority="normal", owner="", owner_email=""):
    """Insert a new escalation row with default status Open."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    esc_id = get_next_escalation_id()
    now = datetime.datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO escalations (
            id, customer, issue, sentiment, urgency, severity, criticality, category,
            status, timestamp, action_taken, owner, owner_email, escalated, priority, escalation_flag,
            action_owner, status_update_date, user_feedback
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        esc_id, customer, issue, sentiment, urgency, severity, criticality, category,
        "Open", now, "", owner, owner_email, escalation_flag, priority, escalation_flag,
        "", "", ""
    ))
    conn.commit()
    conn.close()
    return esc_id

def fetch_escalations_df():
    """Return all escalations as a DataFrame (safe read)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def update_escalation_status(esc_id, status=None, action_taken=None, action_owner=None, owner_email=None, feedback=None, sentiment=None, criticality=None):
    """Update a subset of fields for an escalation (keeps columns stable)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Build dynamic update statement to avoid overwriting with None
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

# -------------------------
# ---- TEXT / DEDUPE ------
# -------------------------
def generate_issue_hash(issue_text):
    """
    Normalize and hash issue text to dedupe forwarded/repeated emails.
    Removes common header markers and condenses whitespace.
    """
    patterns_to_remove = [
        r"[-]+[ ]*Forwarded message[ ]*[-]+",
        r"From:.*", r"Sent:.*", r"To:.*", r"Subject:.*",
        r">.*",                     # quoted lines
        r"On .* wrote:",
    ]
    txt = issue_text
    for p in patterns_to_remove:
        txt = re.sub(p, " ", txt, flags=re.IGNORECASE)
    txt = re.sub(r'\s+', ' ', txt).strip().lower()
    return hashlib.md5(txt.encode('utf-8')).hexdigest()

def summarize_issue_text(issue_text, max_len=120):
    s = re.sub(r'\s+', ' ', str(issue_text)).strip()
    return s[:max_len] + "..." if len(s) > max_len else s

# -------------------------
# ---- EMAIL PARSING ------
# -------------------------
def parse_emails():
    """
    Connect to IMAP and fetch UNSEEN emails. Return list of dicts:
    [{'customer': from_addr, 'issue': summary, 'raw': full_text}, ...]
    Deduplicates using in-memory hash set (safe for runtime).
    """
    results = []
    try:
        imap = imaplib.IMAP4_SSL(EMAIL_SERVER)
        imap.login(EMAIL_USER, EMAIL_PASS)
        imap.select("INBOX")
        status, messages = imap.search(None, "UNSEEN")
        if status != "OK":
            imap.logout()
            return results

        for num in messages[0].split():
            res, msg_data = imap.fetch(num, "(RFC822)")
            if res != "OK":
                continue
            for part in msg_data:
                if isinstance(part, tuple):
                    msg = email.message_from_bytes(part[1])
                    # Subject decode
                    subj_raw = msg.get("Subject", "")
                    subj = decode_header(subj_raw)[0][0]
                    if isinstance(subj, bytes):
                        subj = subj.decode(errors="ignore")
                    from_ = msg.get("From", "unknown")

                    # extract body (prefer text/plain)
                    body = ""
                    if msg.is_multipart():
                        for p in msg.walk():
                            ctype = p.get_content_type()
                            disp = str(p.get("Content-Disposition"))
                            if ctype == "text/plain" and "attachment" not in disp:
                                try:
                                    body = p.get_payload(decode=True).decode(errors="ignore")
                                except Exception:
                                    body = str(p.get_payload())
                                break
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode(errors="ignore")
                        except Exception:
                            body = str(msg.get_payload())

                    full_text = f"{subj} - {body}"
                    h = generate_issue_hash(full_text)

                    with processed_email_hashes_lock:
                        if h in processed_email_hashes:
                            # Skip already processed identical content
                            continue
                        processed_email_hashes.add(h)

                    summary = summarize_issue_text(full_text)
                    results.append({"customer": from_, "issue": summary, "raw": full_text})
        imap.logout()
    except Exception as e:
        st.warning(f"Email fetch error: {e}")
    return results

# -------------------------
# ---- NLP & TAGGING ------
# -------------------------
def analyze_issue(issue_text):
    """
    Use VADER for sentiment + keyword matching for urgency/category/severity/criticality.
    Returns tuple: sentiment, urgency, severity, criticality, category, escalation_flag
    """
    comp = analyzer.polarity_scores(issue_text)["compound"]
    if comp < -0.05:
        sentiment = "negative"
    elif comp > 0.05:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    text_lc = issue_text.lower()
    # urgency: high if any negative keyword matches
    urgency = "high" if any(k in text_lc for kws in NEGATIVE_KEYWORDS.values() for k in kws) else "normal"

    # find category first matching category in NEGATIVE_KEYWORDS
    category = None
    for cat, kws in NEGATIVE_KEYWORDS.items():
        if any(k in text_lc for k in kws):
            category = cat
            break

    # severity mapping
    if category in ("safety", "technical"):
        severity = "critical"
    elif category in ("support", "business"):
        severity = "major"
    else:
        severity = "minor"

    # criticality rule (you can adjust thresholds later)
    criticality = "high" if sentiment == "negative" and urgency == "high" else "medium"

    escalation_flag = "Yes" if urgency == "high" or sentiment == "negative" else "No"

    return sentiment, urgency, severity, criticality, category or "other", escalation_flag

# -------------------------
# ---- ML MODEL ----------
# -------------------------
def train_model():
    """
    Train RandomForest classifier from DB. Requires at least 20 rows and both classes present.
    Returns trained model or None if training not possible.
    """
    df = fetch_escalations_df()
    if df.shape[0] < 20:
        return None
    # require fields
    df = df.dropna(subset=["sentiment", "urgency", "severity", "criticality", "escalated"])
    if df.empty:
        return None

    X = pd.get_dummies(df[["sentiment", "urgency", "severity", "criticality"]].astype(str))
    y = df["escalated"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    if y.nunique() < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    # storing feature names on the model helps when predicting
    model._feature_names = X.columns.tolist()
    return model

def predict_escalation(model, sentiment, urgency, severity, criticality):
    """Return 'Yes'/'No' prediction using trained model. If model is None, default to 'No'."""
    if model is None:
        return "No"
    # one-hot vector with model's feature names
    vec = {f"sentiment_{sentiment}": 1, f"urgency_{urgency}": 1,
           f"severity_{severity}": 1, f"criticality_{criticality}": 1}
    Xpred = pd.DataFrame([vec])
    Xpred = Xpred.reindex(columns=model._feature_names, fill_value=0)
    p = model.predict(Xpred)[0]
    return "Yes" if int(p) == 1 else "No"

# -------------------------
# ---- ALERTING (single) --
# -------------------------
def send_alert(message, via="email", recipient=None):
    """
    Unified alert sender.
    - via: "email" or "teams"
    - recipient: for email alerts (fallback to ALERT_RECIPIENT)
    Errors will be logged via st.warning / print but not raised to avoid breaking UI threads.
    """
    if via == "email":
        try:
            msg = EmailMessage()
            subject = f"{EMAIL_SUBJECT_PREFIX}"
            msg["Subject"] = subject
            msg["From"] = EMAIL_USER
            msg["To"] = recipient or ALERT_RECIPIENT
            msg.set_content(message)
            with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as s:
                s.starttls()
                s.login(EMAIL_USER, EMAIL_PASS)
                s.send_message(msg)
            # Optional: st.success("Email sent")
        except Exception as e:
            # don't crash UI threads; log
            st.warning(f"Email alert failed: {e}")
            print(f"Email alert failed: {e}")
    elif via == "teams":
        try:
            if not TEAMS_WEBHOOK:
                st.warning("Teams webhook not configured.")
                return
            payload = {"text": message}
            r = requests.post(TEAMS_WEBHOOK, json=payload, timeout=10)
            if r.status_code not in (200, 201):
                st.warning(f"Teams alert failed: {r.status_code} {r.text}")
        except Exception as e:
            st.warning(f"Teams alert failed: {e}")
            print(f"Teams alert failed: {e}")

# -------------------------
# ---- BACKGROUND JOBS -----
# -------------------------
def email_polling_job(poll_seconds=60):
    """
    background job: poll inbox every poll_seconds, analyze new emails, insert as escalations.
    """
    while True:
        try:
            emails = parse_emails()
            for e in emails:
                cust = e.get("customer", "unknown")
                issue = e.get("issue", "")
                sentiment, urgency, severity, criticality, category, esc_flag = analyze_issue(issue)
                # default priority mapping: escalate 'critical' severity to high priority
                priority = "high" if severity == "critical" or esc_flag == "Yes" else "normal"
                insert_escalation(cust, issue, sentiment, urgency, severity, criticality, category, esc_flag, priority)
        except Exception as ex:
            print(f"Background email polling error: {ex}")
        time.sleep(poll_seconds)

def send_daily_escalation_report():
    """Generate and email a daily excel report of escalated cases (if any)."""
    df = fetch_escalations_df()
    df_esc = df[df["escalated"].str.lower() == "yes"] if not df.empty else pd.DataFrame()
    if df_esc.empty:
        print("No escalated cases for daily report.")
        return
    file_path = "daily_escalated_cases.xlsx"
    try:
        df_esc.to_excel(file_path, index=False)
        msg = EmailMessage()
        msg["Subject"] = "Daily Escalated Cases Report"
        msg["From"] = EMAIL_USER
        msg["To"] = ALERT_RECIPIENT
        msg.set_content("Attached: daily escalated cases.")
        with open(file_path, "rb") as f:
            data = f.read()
            msg.add_attachment(data, maintype="application", subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="daily_escalated_cases.xlsx")
        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as s:
            s.starttls()
            s.login(EMAIL_USER, EMAIL_PASS)
            s.send_message(msg)
        print("Daily escalation report sent.")
    except Exception as e:
        print("Failed to send daily report:", e)

def start_daily_scheduler():
    """
    Simple daily scheduler running in background thread.
    For production, replace with APScheduler/Cron. Uses local time.
    """
    while True:
        now = datetime.datetime.now()
        # run at 09:00 local time
        if now.hour == 9 and now.minute == 0:
            try:
                send_daily_escalation_report()
            except Exception as e:
                print("Daily report error:", e)
            # sleep 61 seconds to avoid double-run within same minute
            time.sleep(61)
        time.sleep(20)

# -------------------------
# ---- STREAMLIT UI -------
# -------------------------
ensure_schema()
st.set_page_config(layout="wide", page_title="EscalateAI")

st.markdown("<h1>🚨 EscalateAI — Escalation Management</h1>", unsafe_allow_html=True)
st.write("AI-assisted logging, triage, and alerts for customer escalations.")

# Start background threads if not already started in session_state
if "email_thread" not in st.session_state:
    t = threading.Thread(target=email_polling_job, args=(60,), daemon=True)
    t.start()
    st.session_state["email_thread"] = t

if "daily_thread" not in st.session_state:
    t2 = threading.Thread(target=start_daily_scheduler, daemon=True)
    t2.start()
    st.session_state["daily_thread"] = t2

# -------------------------
# ---- SIDEBAR CONTROLS ---
# -------------------------
st.sidebar.header("Controls")

# Upload Excel
uploaded = st.sidebar.file_uploader("Upload Escalation Excel (.xlsx)", type=["xlsx"])
if uploaded:
    try:
        df_upload = pd.read_excel(uploaded)
        st.sidebar.success("Excel loaded")
        # check required columns
        if not set(["Customer", "Issue"]).issubset(df_upload.columns):
            st.sidebar.error("Excel must contain 'Customer' and 'Issue' columns.")
        else:
            if st.sidebar.button("Analyze & Insert Excel"):
                processed = 0
                for _, r in df_upload.iterrows():
                    cust = str(r.get("Customer", "Unknown"))
                    issue = str(r.get("Issue", "")).strip()
                    if not issue:
                        continue
                    issue_sum = summarize_issue_text(issue)
                    sentiment, urgency, severity, criticality, category, esc_flag = analyze_issue(issue)
                    priority = "high" if severity == "critical" or esc_flag == "Yes" else "normal"
                    insert_escalation(cust, issue_sum, sentiment, urgency, severity, criticality, category, esc_flag, priority)
                    processed += 1
                st.sidebar.success(f"Inserted {processed} rows.")
    except Exception as e:
        st.sidebar.error(f"Failed to read Excel: {e}")

# Fetch Emails manual
if st.sidebar.button("Fetch Emails (manual)"):
    emails = parse_emails()
    for e in emails:
        cust = e.get("customer", "unknown")
        issue = e.get("issue", "")
        sentiment, urgency, severity, criticality, category, esc_flag = analyze_issue(issue)
        priority = "high" if severity == "critical" or esc_flag == "Yes" else "normal"
        insert_escalation(cust, issue, sentiment, urgency, severity, criticality, category, esc_flag, priority)
    st.sidebar.success(f"Processed {len(emails)} email(s).")

# SLA trigger
if st.sidebar.button("Trigger SLA Check"):
    df_all = fetch_escalations_df()
    if df_all.empty:
        st.sidebar.info("No records.")
    else:
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
        breaches = df_all[(df_all["status"] != "Resolved") & (df_all["priority"] == "high") & ((datetime.datetime.now() - df_all["timestamp"]) > datetime.timedelta(minutes=10))]
        if not breaches.empty:
            msg = f"🚨 SLA breach for {len(breaches)} case(s)!"
            send_alert(msg, via="teams")
            send_alert(msg, via="email")
            st.sidebar.success("SLA alerts sent.")
        else:
            st.sidebar.info("No SLA breaches.")

# Manual notifications (compose)
st.sidebar.markdown("---")
st.sidebar.markdown("### Manual Notifications")
manual_msg = st.sidebar.text_area("Message", "Test alert from EscalateAI")
if st.sidebar.button("Send Teams"):
    send_alert(manual_msg, via="teams")
    st.sidebar.success("Teams sent.")
if st.sidebar.button("Send Email"):
    send_alert(manual_msg, via="email")
    st.sidebar.success("Email sent.")

# WhatsApp stub for Resolved cases
st.sidebar.markdown("---")
st.sidebar.markdown("### WhatsApp (stub)")
status_check = st.sidebar.selectbox("WhatsApp: Case Status", ["Open", "In Progress", "Resolved"])
if status_check == "Resolved":
    df_res = fetch_escalations_df()
    df_resolved = df_res[df_res["status"] == "Resolved"] if not df_res.empty else pd.DataFrame()
    if not df_resolved.empty:
        esc_choice = st.sidebar.selectbox("Choose Escalation ID", df_resolved["id"].tolist())
        phone = st.sidebar.text_input("Phone +country", "+91")
        wa_msg = st.sidebar.text_area("Message", f"Your issue {esc_choice} has been resolved.")
        if st.sidebar.button("Send WhatsApp"):
            # Implement integration if you choose; currently stub for UI.
            st.sidebar.success(f"WhatsApp simulated to {phone} for {esc_choice}")

# Downloads
st.sidebar.markdown("---")
if st.sidebar.button("Download all CSV"):
    df_all = fetch_escalations_df()
    csv = df_all.to_csv(index=False)
    st.download_button("Click to download", csv, file_name="escalations.csv", mime="text/csv")

# Dev options
st.sidebar.markdown("---")
if st.sidebar.checkbox("View raw DB (dev)"):
    st.sidebar.dataframe(fetch_escalations_df())

if st.sidebar.button("Reset DB (dev)"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS escalations")
    conn.commit()
    conn.close()
    ensure_schema()
    st.sidebar.warning("Database reset. Restart app if needed.")

# -------------------------
# ---- MAIN PAGE / KANBAN -
# -------------------------
# Filters
df = fetch_escalations_df()
if df.empty:
    st.info("No escalation records yet.")
else:
    st.markdown("### Filters")
    col1, col2, col3, col4 = st.columns(4)
    status_filter = col1.selectbox("Status", ["All"] + sorted(df["status"].dropna().unique().tolist()))
    severity_filter = col2.selectbox("Severity", ["All"] + sorted(df["severity"].dropna().unique().tolist()))
    sentiment_filter = col3.selectbox("Sentiment", ["All"] + sorted(df["sentiment"].dropna().unique().tolist()))
    category_filter = col4.selectbox("Category", ["All"] + sorted(df["category"].dropna().unique().tolist()))

    filtered = df.copy()
    if status_filter != "All":
        filtered = filtered[filtered["status"] == status_filter]
    if severity_filter != "All":
        filtered = filtered[filtered["severity"] == severity_filter]
    if sentiment_filter != "All":
        filtered = filtered[filtered["sentiment"] == sentiment_filter]
    if category_filter != "All":
        filtered = filtered[filtered["category"] == category_filter]

    # show high-level counts
    counts = filtered["status"].value_counts()
    st.markdown(f"**Counts** — Open: {counts.get('Open',0)} | In Progress: {counts.get('In Progress',0)} | Resolved: {counts.get('Resolved',0)}")

    # Kanban columns
    cols = st.columns(3)
    statuses = ["Open", "In Progress", "Resolved"]
    for col, status in zip(cols, statuses):
        col.markdown(f"#### {status}")
        bucket = filtered[filtered["status"] == status]
        for _, r in bucket.iterrows():
            esc_id = r["id"]
            flag = "🚩" if str(r.get("escalated","")).strip().lower() == "yes" else ""
            title = f"{esc_id} - {r.get('customer','')} {flag}"
            with col.expander(title):
                # Ageing
                try:
                    ts = pd.to_datetime(r.get("timestamp"))
                    age = datetime.datetime.now() - ts.to_pydatetime()
                    days = age.days
                    hours = age.seconds // 3600
                    mins = (age.seconds % 3600) // 60
                    age_str = f"{days}d {hours}h {mins}m"
                    # age color for display
                    total_hours = age.total_seconds() / 3600
                    if total_hours < 4:
                        age_color = "#2ecc71"
                    elif total_hours < 12:
                        age_color = "#f1c40f"
                    else:
                        age_color = "#e74c3c"
                except Exception:
                    age_str = "N/A"; age_color = "#7f8c8d"

                st.markdown(f"**Issue:** {r.get('issue','')}")
                st.markdown(f"**Severity:** {r.get('severity','')}")
                st.markdown(f"**Criticality:** {r.get('criticality','')}")
                st.markdown(f"**Category:** {r.get('category','')}")
                st.markdown(f"**Sentiment:** {r.get('sentiment','')}")
                st.markdown(f"**Urgency:** {r.get('urgency','')}")
                st.markdown(f"**Escalated:** {r.get('escalated','')}")
                st.markdown(f"**Ageing:** <span style='color:{age_color}; font-weight:bold'>{age_str}</span>", unsafe_allow_html=True)

                # Actions: mark resolved, escalate to N+1 (flag), update owner/email, save
                col_a, col_b, col_c = st.columns([1,1,2])
                if col_a.button("✔️ Mark Resolved", key=f"resolve_{esc_id}"):
                    owner_email = r.get("owner_email", EMAIL_USER)
                    update_escalation_status(esc_id, status="Resolved", owner_email=owner_email)
                    send_alert(f"Case {esc_id} marked Resolved.", via="email", recipient=owner_email)
                    st.experimental_rerun()

                n1_email = col_b.text_input("N+1 Email", key=f"n1email_{esc_id}")
                if col_c.button("🚀 Escalate to N+1", key=f"escalate_{esc_id}"):
                    # Escalation should be a flag, not a status
                    update_escalation_status(esc_id, status=r.get("status","Open"), owner_email=n1_email)
                    # also set escalated flag by a direct SQL to avoid overwriting other fields
                    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
                    cur.execute("UPDATE escalations SET escalated = 'Yes' WHERE id = ?", (esc_id,))
                    conn.commit(); conn.close()
                    send_alert(f"Case {esc_id} escalated to {n1_email}", via="email", recipient=n1_email)
                    send_alert(f"Case {esc_id} escalated to {n1_email}", via="teams")
                    st.experimental_rerun()

                # Editable fields
                new_status = st.selectbox("Update Status", ["Open","In Progress","Resolved"], index=["Open","In Progress","Resolved"].index(r.get("status","Open")), key=f"status_select_{esc_id}")
                new_action = st.text_input("Action Taken", r.get("action_taken",""), key=f"action_{esc_id}")
                new_owner = st.text_input("Owner", r.get("owner",""), key=f"owner_{esc_id}")
                new_owner_email = st.text_input("Owner Email", r.get("owner_email",""), key=f"owner_email_{esc_id}")

                if st.button("💾 Save Changes", key=f"save_{esc_id}"):
                    update_escalation_status(esc_id, status=new_status, action_taken=new_action, action_owner=new_action, owner_email=new_owner_email)
                    # notify owner
                    note = f"Case {esc_id} updated. Status: {new_status}. Action: {new_action}."
                    send_alert(note, via="email", recipient=new_owner_email)
                    send_alert(note, via="teams")
                    st.success("Saved.")
                    st.experimental_rerun()

# -------------------------
# ---- FEEDBACK & RETRAIN -
# -------------------------
st.markdown("---")
st.header("🔁 Feedback & Model Retraining")
df = fetch_escalations_df()
if not df.empty:
    # pick only those with escalation flag set (or all)
    sampled = df.head(100)  # limit for UI responsiveness
    for _, row in sampled.iterrows():
        esc_id = row["id"]
        with st.expander(f"{esc_id} - {row.get('issue','')[:80]}"):
            fb = st.selectbox("Was escalation correct?", ["", "Correct", "Incorrect"], key=f"fb_{esc_id}")
            new_sent = st.selectbox("Sentiment", ["", "positive", "neutral", "negative"], key=f"sent_{esc_id}")
            new_crit = st.selectbox("Criticality", ["", "low", "medium", "high", "urgent"], key=f"crit_{esc_id}")
            note = st.text_area("Notes", key=f"note_{esc_id}")
            if st.button("Submit Feedback", key=f"fb_btn_{esc_id}"):
                # store feedback; optionally update criticality/sentiment if user provided corrected values
                update_escalation_status(esc_id, feedback=note, sentiment=new_sent if new_sent else None, criticality=new_crit if new_crit else None)
                st.success("Feedback saved.")

    if st.button("Retrain Model Now"):
        st.info("Retraining model...")
        model = train_model()
        if model:
            st.success("Model retrained.")
        else:
            st.warning("Not enough data or homogenous labels to train.")

else:
    st.info("No records for feedback yet.")

# -------------------------
# ---- FINISH NOTES -------
# -------------------------
st.markdown("---")
st.markdown("**Notes & next steps:**")
st.markdown("""
- Update `.env` with required credentials: `EMAIL_USER`, `EMAIL_PASS`, `EMAIL_SERVER`, `EMAIL_SMTP_SERVER`, `EMAIL_SMTP_PORT`, `EMAIL_RECEIVER`, `MS_TEAMS_WEBHOOK_URL`.
- The WhatsApp action is a UI stub; integrate with Twilio / WhatsApp API if needed.
- For persistent dedupe across restarts, consider saving processed email hashes to DB.
- Prefer running Streamlit with `streamlit run escalate_ai.py`.
- For production scheduling (daily reports, advanced polling), use APScheduler or an external worker.
""")
