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

# Import your existing functions
from EscalateAIV610082025 import fetch_escalations, train_model

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

# 📄 PDF Report Generation
def generate_pdf_report():
    df = fetch_escalations()
    html = df.to_html(index=False)
    pdfkit.from_string(html, "report.pdf")

# 🔥 SLA Heatmap Visualization
def render_sla_heatmap():
    df = fetch_escalations()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    heatmap_data = df.pivot_table(index='category', columns='hour', values='id', aggfunc='count').fillna(0)
    st.subheader("🔥 SLA Breach Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, ax=ax, cmap="Reds")
    st.pyplot(fig)

# 🌙 Dark Mode Toggle
def apply_dark_mode():
    st.markdown("""
    <style>
    body { background-color: #121212; color: #e0e0e0; }
    .sidebar .sidebar-content { background-color: #1e1e1e; }
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
    return f"🔎 Summary: {total} total cases, {escalated} escalated."
