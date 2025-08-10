# escalateai_enhancements.py

import pandas as pd
import altair as alt
import pdfkit
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

nlp = spacy.load("en_core_web_sm")

# === 1. Auto-tagging ===
def extract_tags(issue_text):
    """Extract named entities from issue text using spaCy."""
    doc = nlp(issue_text)
    return [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG", "GPE", "DATE"]]

# === 2. Feature importance ===
def show_feature_importance(model, st):
    """Display top features influencing escalation prediction."""
    importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    st.subheader("🧠 Top Features Influencing Escalation Decisions")
    st.bar_chart(importances.sort_values(ascending=False).head(5))

# === 3. Feedback-weighted model training ===
def train_model(df):
    """Train RandomForest model using feedback accuracy as sample weights."""
    if df.shape[0] < 20:
        return None
    df = df.dropna(subset=['sentiment', 'urgency', 'severity', 'criticality', 'escalated'])
    if df.empty:
        return None
    X = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
    y = df['escalated'].apply(lambda x: 1 if x == 'Yes' else 0)
    weights = df.get('feedback_accuracy', pd.Series([1]*len(df)))
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    return model

# === 4. Escalation trend chart ===
def escalation_trend_chart(df, st):
    """Render escalation volume over time using Altair."""
    df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
    chart = alt.Chart(df).mark_bar().encode(
        x='date:T',
        y='count():Q',
        color='severity:N'
    ).properties(title="📊 Escalation Volume Over Time")
    st.altair_chart(chart, use_container_width=True)

# === 5. PDF report generator ===
def generate_pdf_report(df, st):
    """Generate and offer download of escalation report as PDF."""
    html = df.to_html(index=False)
    pdfkit.from_string(html, "escalation_report.pdf")
    with open("escalation_report.pdf", "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name="escalation_report.pdf", mime="application/pdf")

# === 6. Dark mode toggle ===
def enable_dark_mode(st):
    """Inject dark mode CSS if enabled."""
    st.markdown("""
    <style>
    body { background-color: #121212; color: #e0e0e0; }
    .stButton button { background-color: #333; color: white; }
    </style>
    """, unsafe_allow_html=True)

# === 7. Sticky filter summary ===
def sticky_filter_summary(st, status, severity, sentiment, category):
    """Display sticky filter summary bar."""
    st.markdown(f"""
    <div style='position:sticky;top:0;background:#f8f9fa;padding:8px;border-bottom:1px solid #ccc;'>
    <b>Filters:</b> Status = {status} | Severity = {severity} | Sentiment = {sentiment} | Category = {category}
    </div>
    """, unsafe_allow_html=True)
