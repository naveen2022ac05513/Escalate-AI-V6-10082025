import pandas as pd
import datetime
import schedule
import time
import threading
import plotly.express as px
import plotly.io as pio
from rapidfuzz import fuzz
import streamlit as st
import sqlite3
from xhtml2pdf import pisa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

pio.kaleido.scope.default_format = "png"
DB_PATH = "escalations.db"

# 🔁 Fetch escalation data
def fetch_escalations():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

# 🔄 Auto-Retraining Scheduler
def schedule_weekly_retraining():
    schedule.every().sunday.at("09:00").do(train_model)
    threading.Thread(target=run_scheduler, daemon=True).start()

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

# 🧠 Model Training
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

# 📊 Category Breakdown Chart
def generate_category_chart():
    df = fetch_escalations()
    if df.empty or "category" not in df.columns:
        st.warning("No category data available.")
        return px.bar(title="No Data")
    category_counts = df["category"].value_counts().reset_index()
    category_counts.columns = ["Category", "Count"]
    fig = px.bar(category_counts, x="Category", y="Count", title="Category Breakdown")
    return fig

def render_category_breakdown():
    fig = generate_category_chart()
    fig.write_html("category_chart.html")
    with open("category_chart.html", "r", encoding="utf-8") as f:
        st.download_button(
            label="📥 Download Category Chart (HTML)",
            data=f.read(),
            file_name="category_chart.html",
            mime="text/html"
        )
    st.plotly_chart(fig, use_container_width=True)

# ⏰ SLA Breach Trend
def render_sla_trend():
    df = fetch_escalations()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['Day'] = df['timestamp'].dt.date
    now = datetime.datetime.now()
    df["SLA_Breach_Flag"] = (
        (df["status"] != "Resolved") &
        (df["priority"] == "high") &
        ((now - df["timestamp"]) > datetime.timedelta(minutes=10))
    )
    breach_trend = df[df["SLA_Breach_Flag"]].groupby("Day").size().reset_index(name="Breaches")
    if breach_trend.empty:
        st.info("✅ No SLA breaches to show.")
        return
    st.subheader("⏰ SLA Breach Trend")
    fig = px.line(breach_trend, x="Day", y="Breaches", title="SLA Breaches Over Time", markers=True)
    st.plotly_chart(fig)
    st.download_button(
        label="📥 Download SLA Trend (HTML)",
        data=fig.to_html(),
        file_name="sla_trend.html",
        mime="text/html"
    )

# 📊 Full Dashboard
def render_full_analytics_dashboard():
    st.subheader("📊 Category Breakdown")
    render_category_breakdown()
    st.subheader("📈 SLA Trend")
    render_sla_trend()
