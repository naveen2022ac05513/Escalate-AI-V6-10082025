# enhancement_dashboard.py

import streamlit as st
from enhancements import fetch_escalations, train_model
from advanced_enhancements import generate_shap_plot, generate_pdf_report

def show_enhancement_dashboard():
    st.title("🧠 Enhancement Dashboard")

    st.markdown("Use this module to analyze escalations, generate SHAP plots, and download reports.")

    # Load data
    escalations = fetch_escalations()
    st.subheader("📋 Escalation Data")
    st.dataframe(escalations)

    # Model training
    st.subheader("⚙️ Train Model")
    if st.button("Train"):
        model, X_test, y_test = train_model(escalations)
        st.success("Model trained successfully.")

        # SHAP plot
        st.subheader("📊 SHAP Plot")
        shap_fig = generate_shap_plot(model, X_test)
        st.pyplot(shap_fig)

        # PDF report
        st.subheader("📄 Generate PDF Report")
        if st.button("Download Report"):
            pdf_bytes = generate_pdf_report(escalations)
            st.download_button("📥 Download PDF", data=pdf_bytes, file_name="enhancement_report.pdf")
