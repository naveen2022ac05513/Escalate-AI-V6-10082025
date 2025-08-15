import streamlit as st
import numpy as np
from advanced_enhancements import (
    train_model,
    fetch_escalations,
    generate_shap_plot,
    render_model_metrics
)

def show_enhancement_dashboard():
    st.title("📈 Enhancement Dashboard")

    escalations = fetch_escalations()

    debug = st.sidebar.checkbox("🛠️ Show Raw Escalation Data")
    if debug:
        st.subheader("🔍 Raw Escalation Data")
        st.dataframe(escalations)

    # 🧪 Inject synthetic 'is_escalation' column if missing (for testing)
    if 'is_escalation' not in escalations.columns:
        st.warning("⚠️ Simulating 'is_escalation' column for testing.")
        escalations['is_escalation'] = np.random.choice([0, 1], size=len(escalations))

    try:
        model, X_test, y_test = train_model(escalations)
        st.success("✅ Model trained successfully.")

        show_model_insights(model, X_test, y_test)

    except KeyError as e:
        st.error(f"🚫 Enhancement dashboard unavailable: {e}")
        show_fallback_dashboard(escalations)

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.exception(e)

def show_model_insights(model, X_test, y_test):
    st.subheader("📊 Model Insights")

    try:
        accuracy = model.score(X_test, y_test)
        st.metric("Model Accuracy", f"{accuracy:.2%}")

        generate_shap_plot(model, X_test)
        render_model_metrics(model, X_test, y_test)

    except Exception as e:
        st.warning("⚠️ Unable to display model insights.")
        st.exception(e)

def show_fallback_dashboard(escalations):
    st.subheader("🧪 Fallback View: Diagnostics & Metadata")

    if escalations.empty:
        st.warning("No escalation data available.")
        return

    st.write("Available columns:", list(escalations.columns))
    st.dataframe(escalations.head())

    if 'created_at' in escalations.columns:
        st.line_chart(escalations['created_at'].value_counts().sort_index())

    st.info("ℹ️ Add 'is_escalation' column to enable model training.")
