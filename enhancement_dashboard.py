import streamlit as st
from advanced_enhancements import train_model, fetch_escalations

def show_enhancement_dashboard():
    st.title("📈 Enhancement Dashboard")

    escalations = fetch_escalations()

    debug = st.sidebar.checkbox("🛠️ Show Raw Escalation Data")
    if debug:
        st.subheader("🔍 Raw Escalation Data")
        st.dataframe(escalations)

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

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            st.bar_chart(importances)

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
