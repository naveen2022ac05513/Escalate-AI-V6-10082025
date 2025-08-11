# enhancements.py

import pandas as pd
import datetime
import schedule
import time
import threading
import plotly.express as px
from rapidfuzz import fuzz
import streamlit as st
import pdfkit
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DB_PATH = "escalations.db"

# 🔄 Auto-Retraining Scheduler
def schedule_weekly_retraining():
    schedule.every().sunday.at("09:00").do(train_model)
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)
    threading.Thread(target=run_scheduler, daemon=True).start()

# 📊 Interactive Analytics Dashboard
def render_analytics():
    df = fetch_escalations()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    st.subheader("📊 Escalation Trends")
    st.plotly_chart(px.histogram(df, x="timestamp", color="severity", title="Escalations Over Time"))
    st.plotly_chart(px.pie(df, names="sentiment", title="Sentiment Distribution"))

# 🧠 Explainable ML (Feature Importance)
def show_feature_importance(model):
    importance = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    st.subheader("🧠 Feature Importance")
    st.plotly_chart(px.bar(importance.sort_values(), orientation='h', title="Top Predictive Features"))

# 🧪 Fuzzy Deduplication
def is_duplicate(issue_text, threshold=90):
    df = fetch_escalations()
    for existing in df["issue"]:
        if fuzz.partial_ratio(issue_text, existing) > threshold:
            return True
    return False

# PDF Generator

from xhtml2pdf import pisa
import pandas as pd

def generate_pdf_report():
    df = fetch_escalations()
    html = f"""
    <html>
    <head>
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            h2 {{
                text-align: center;
                color: #2c3e50;
            }}
        </style>
    </head>
    <body>
        <h2>📄 Escalation Report</h2>
        {df.to_html(index=False)}
    </body>
    </html>
    """
    try:
        with open("report.pdf", "wb") as f:
            pisa.CreatePDF(html, dest=f)
        print("✅ PDF report generated successfully.")
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")

# 🔥 SLA Heatmap Visualization
def render_sla_heatmap():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st

    # Fetch escalation data
    df = fetch_escalations()

    if df.empty or "timestamp" not in df.columns or "priority" not in df.columns:
        st.warning("Insufficient data to render SLA heatmap.")
        return

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Create a 'Day' column for grouping
    df['Day'] = df['timestamp'].dt.date

    # Optional: Assign teams if not present
    if "Team" not in df.columns:
        df["Team"] = df["owner"].fillna("Unassigned")

    # Define SLA breach condition (e.g., unresolved high-priority cases older than 10 minutes)
    now = datetime.datetime.now()
    df["SLA_Breach_Flag"] = (
        (df["status"] != "Resolved") &
        (df["priority"] == "high") &
        ((now - df["timestamp"]) > datetime.timedelta(minutes=10))
    )

    # Aggregate breach counts
    heatmap_data = df[df["SLA_Breach_Flag"]].pivot_table(
        index="Team",
        columns="Day",
        values="SLA_Breach_Flag",
        aggfunc="sum",
        fill_value=0
    )

    if heatmap_data.empty:
        st.info("✅ No SLA breaches to display.")
        return

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="Reds",
        linewidths=0.5,
        linecolor='white',
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "SLA Breach Count"}
    )
    ax.set_title("🔥 SLA Breach Heatmap", fontsize=14)
    st.pyplot(fig)
    

# 🌙 Dark Mode Toggle
def apply_dark_mode():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #121212 !important;
        color: #e0e0e0 !important;
    }
    .stApp {
        background-color: #121212 !important;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e !important;
    }
    </style>
    """, unsafe_allow_html=True)


# 📌 Sticky Filter Summary
def show_filter_summary(status, severity, sentiment, category):
    st.sidebar.markdown(f"""
    <div style='position:sticky;top:10px;background:#f0f0f0;padding:6px;border-radius:5px'>
    <b>Filters:</b><br>
    Status: {status}<br>
    Severity: {severity}<br>
    Sentiment: {sentiment}<br>
    Category: {category}
    </div>
    """, unsafe_allow_html=True)

# 📧 Escalation Message Templates
def get_escalation_template(severity):
    TEMPLATES = {
        "critical": "🚨 Immediate action required for critical issue.",
        "major": "⚠️ Major issue reported. Please investigate.",
        "minor": "ℹ️ Minor issue logged for review."
    }
    return TEMPLATES.get(severity.lower(), "🔔 New escalation update.")

# 🧠 AI Assistant Summary
def summarize_escalations():
    df = fetch_escalations()
    total = len(df)
    escalated = df[df['escalated'] == 'Yes'].shape[0]
    return f"🔎 Summary: 📌Total cases: {total},🚨 Escalated: {escalated}."

# 🔁 Local copy of fetch_escalations
def fetch_escalations():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

# 🔁 Local copy of train_model
def train_model():
    df = fetch_escalations()
    if df.shape[0] < 20:
        return None
    df = df.dropna(subset=['sentiment', 'urgency', 'severity', 'criticality', 'escalated'])
    if df.empty:
        return None
    X = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
    y = df['escalated'].apply(lambda x: 1 if x == 'Yes' else 0)
    if y.nunique() < 2:
        return None
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
