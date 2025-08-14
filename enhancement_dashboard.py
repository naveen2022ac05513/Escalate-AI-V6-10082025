import streamlit as st
import pandas as pd
from advanced_enhancements import (
    train_model, 
    generate_shap_plot, 
    generate_pdf_report, 
    fetch_escalations
)

import streamlit as st
from advanced_enhancements import train_model

def show_enhancement_dashboard():
    st.title("📈 Enhancement Dashboard")

    try:
        from advanced_enhancements import fetch_escalations
        escalations = fetch_escalations()
        # escalations = load_escalation_data()  # however you load it
        model, X_test, y_test = train_model(escalations)
        st.success("Model trained successfully.")
        show_model_insights(model, X_test, y_test)

    except KeyError as e:
        st.error(f"🚫 Enhancement dashboard unavailable: {e}")
        show_fallback_dashboard(escalations)

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()
        
    # Train model
    if st.button("Train Model"):
        model, X_test, y_test = train_model(escalations)
        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.success("✅ Model trained successfully.")

    # Show SHAP plot
    if "model" in st.session_state and "X_test" in st.session_state:
        st.subheader("🔍 SHAP Summary Plot")
        generate_shap_plot(st.session_state.model, st.session_state.X_test)
    else:
        st.info("ℹ️ Train the model to view SHAP explanations.")

    # Generate PDF report
    if st.button("Generate PDF Report"):
        generate_pdf_report()
        st.success("📄 PDF report generated.")

    # Optional: Add download button if needed
    try:
        with open("report.pdf", "rb") as f:
            st.download_button("📥 Download Report", f, file_name="enhancement_report.pdf")
    except FileNotFoundError:
        st.info("ℹ️ No report available yet. Please generate one first.")

def show_analytics_view():
    st.title("📊 Escalation Analytics")

    df = fetch_escalations()

    if df.empty:
        st.warning("⚠️ No escalation data available.")
        return

    # 📈 Trend Over Time
    st.subheader("📈 Escalation Volume Over Time")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    trend = df.groupby(df['timestamp'].dt.date).size()
    st.line_chart(trend)

    # 🔥 Severity Distribution
    st.subheader("🔥 Severity Distribution")
    st.bar_chart(df['severity'].value_counts())

    # 🧠 Sentiment Breakdown
    st.subheader("🧠 Sentiment Breakdown")
    st.bar_chart(df['sentiment'].value_counts())

    # ⏳ Ageing Buckets
    st.subheader("⏳ Ageing Buckets")
    df['age_days'] = (pd.Timestamp.now() - df['timestamp']).dt.days
    bins = [0, 3, 7, 14, 30, 90]
    labels = ["0–3d", "4–7d", "8–14d", "15–30d", "31–90d"]
    df['age_bucket'] = pd.cut(df['age_days'], bins=bins, labels=labels)
    st.bar_chart(df['age_bucket'].value_counts().sort_index())


    # Fallback Dashboard
def show_fallback_dashboard(escalations):
    st.subheader("🔍 Fallback View: Diagnostics & Metadata")

    if escalations.empty:
        st.warning("No escalation data available.")
        return

    st.write("Available columns:", list(escalations.columns))
    st.dataframe(escalations.head())

    if 'created_at' in escalations.columns:
        st.line_chart(escalations['created_at'].value_counts().sort_index())

    st.info("Add 'is_escalation' column to enable model training.")
