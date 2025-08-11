# enhancements.py (robust version — safe against "no data" and seaborn nanmin crashes)

import pandas as pd
import datetime
import schedule
import time
import threading
import plotly.express as px
from rapidfuzz import fuzz
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from xhtml2pdf import pisa

DB_PATH = "escalations.db"

# ---------------------------
# Utility / scheduler helpers
# ---------------------------
def schedule_weekly_retraining():
    schedule.every().sunday.at("09:00").do(train_model)
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)
    threading.Thread(target=run_scheduler, daemon=True).start()

# ---------------------------
# Data helpers
# ---------------------------
def fetch_escalations():
    """Read escalations table (returns empty DataFrame on error)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

# ---------------------------
# Analytics helpers
# ---------------------------
def render_analytics():
    df = fetch_escalations()
    if df.empty:
        st.warning("⚠️ No escalation data to display analytics.")
        return
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    st.subheader("📊 Escalation Trends")
    st.plotly_chart(px.histogram(df, x="timestamp", color="severity", title="Escalations Over Time"))
    st.plotly_chart(px.pie(df, names="sentiment", title="Sentiment Distribution"))

def show_feature_importance(model):
    importance = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    st.subheader("🧠 Feature Importance")
    st.plotly_chart(px.bar(importance.sort_values(), orientation='h', title="Top Predictive Features"))

# ---------------------------
# Misc helpers
# ---------------------------
def is_duplicate(issue_text, threshold=90):
    df = fetch_escalations()
    if df.empty or 'issue' not in df.columns:
        return False
    for existing in df["issue"].astype(str):
        if fuzz.partial_ratio(issue_text, existing) > threshold:
            return True
    return False

def generate_pdf_report():
    df = fetch_escalations()
    if df.empty:
        st.warning("⚠️ No data available to generate PDF report.")
        return
    html = f"""
    <html><head>
    <style>
      table {{ width: 100%; border-collapse: collapse; }}
      th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
      th {{ background-color: #f2f2f2; }}
      h2 {{ text-align: center; color: #2c3e50; }}
    </style>
    </head><body>
    <h2>📄 Escalation Report</h2>
    {df.to_html(index=False)}
    </body></html>
    """
    try:
        with open("report.pdf", "wb") as f:
            pisa.CreatePDF(html, dest=f)
        st.success("✅ PDF report generated successfully.")
    except Exception as e:
        st.error(f"❌ PDF generation failed: {e}")

# ---------------------------
# SLA heatmap (robust)
# ---------------------------
def render_sla_heatmap():
    """
    Render a heatmap of counts by category (rows) x hour (columns).
    Extremely defensive: returns early for ANY 'no data' condition
    and wraps seaborn plotting in try/except, so the app never crashes.
    """
    df = fetch_escalations()

    # 1) No data at all
    if df.empty:
        st.info("🔥 SLA Heatmap: no escalation records found.")
        return

    # 2) Ensure timestamp + category exist
    if 'timestamp' not in df.columns or 'category' not in df.columns:
        st.warning("⚠️ SLA Heatmap: required columns ('timestamp' or 'category') missing.")
        return

    # 3) Parse timestamps and drop rows missing critical fields
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'category'])

    if df.empty:
        st.info("🔥 SLA Heatmap: no records with valid timestamp/category.")
        return

    # 4) Create hour column and pivot
    df['hour'] = df['timestamp'].dt.hour
    heatmap_data = df.pivot_table(
        index='category',
        columns='hour',
        values='id',
        aggfunc='count'
    )

    # 5) If pivot yields no rows/columns, skip
    if heatmap_data is None or heatmap_data.empty:
        st.info("🔥 SLA Heatmap: pivot produced no data to plot.")
        return

    # 6) Convert to numeric safely (coerce non-numeric to NaN), then fill zeros
    heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')
    # if after coercion it's all NaN -> nothing meaningful
    if heatmap_data.isnull().all().all():
        st.info("🔥 SLA Heatmap: after numeric coercion there are no numeric values.")
        return

    heatmap_data = heatmap_data.fillna(0)

    # 7) Strict numeric checks
    try:
        arr = np.asarray(heatmap_data.values, dtype=float)
    except Exception:
        st.warning("⚠️ SLA Heatmap: unable to convert pivot table to numeric array.")
        return

    if arr.size == 0:
        st.info("🔥 SLA Heatmap: numeric array is empty.")
        return

    if not np.isfinite(arr).any():
        st.info("🔥 SLA Heatmap: numeric array contains no finite values.")
        return

    if (arr == 0).all():
        st.info("🔥 SLA Heatmap: all counts are zero (nothing to visualize).")
        return

    # 8) Finally try plotting in a safe try/except block
    st.subheader("🔥 SLA Breach Heatmap")
    fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * heatmap_data.shape[0])))

    try:
        # pass DataFrame so seaborn can label axes
        sns.heatmap(heatmap_data, ax=ax, cmap="Reds", cbar=True, annot=False)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Category")
        st.pyplot(fig)
    except Exception as e:
        # If seaborn still fails for some unforeseen reason, fallback gracefully
        st.warning(f"⚠️ SLA heatmap rendering skipped due to plotting error.")
        # For debug logs only (prints to server logs, not to UI)
        print(f"[enhancements.render_sla_heatmap] seaborn plotting failed: {e}")
        plt.close(fig)
        return

# ---------------------------
# Basic ML/train helper
# ---------------------------
def train_model():
    df = fetch_escalations()
    if df.shape[0] < 20:
        return None
    # ensure columns exist
    required = ['sentiment', 'urgency', 'severity', 'criticality', 'escalated']
    if not all(col in df.columns for col in required):
        return None
    df = df.dropna(subset=required)
    if df.empty:
        return None
    X = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
    y = df['escalated'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    if y.nunique() < 2:
        return None
    X_train, _, y_train, _ = train_test_split(X, y,
