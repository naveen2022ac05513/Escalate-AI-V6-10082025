import streamlit as st
import pandas as pd
from advanced_enhancements import (
    train_model, 
    generate_shap_plot, 
    generate_pdf_report, 
    fetch_escalations
)

def show_enhancement_dashboard():
    st.title("🚀 Enhancement Dashboard")

    # Load escalation data
    escalations = fetch_escalations()

    if escalations.empty:
        st.warning("⚠️ No escalation data available.")
        return

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
