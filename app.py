import os
import streamlit as st

st.set_page_config(page_title="TruthLens AI", page_icon="🔎", layout="wide")
st.title("🔎 TruthLens AI")
st.subheader("Transformer-Based Fake News Detection")
st.write(
    "Paste a news headline or article. A fine-tuned DistilBERT model "
    "will estimate whether its learned text patterns resemble REAL or FAKE news."
)
st.divider()

if not os.path.exists("model/config.json"):
    st.warning("The trained model is not available yet. Run `python train.py` first.")
    st.stop()

from predict import predict_news

news_text = st.text_area(
    "📰 Enter News Article",
    height=250,
    placeholder="Paste the news headline or complete article here...",
)

if st.button("🔍 Analyze News", use_container_width=True):
    if not news_text.strip():
        st.warning("Please enter a news article.")
    else:
        with st.spinner("Analyzing with DistilBERT..."):
            result = predict_news(news_text)

        st.divider()
        st.subheader("Prediction Result")

        if result["label"] == "FAKE":
            st.error("⚠️ FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")

        st.metric("Confidence", f'{result["confidence"] * 100:.2f}%')

        col1, col2 = st.columns(2)
        with col1:
            st.metric("REAL Probability", f'{result["real_probability"] * 100:.2f}%')
            st.progress(result["real_probability"])
        with col2:
            st.metric("FAKE Probability", f'{result["fake_probability"] * 100:.2f}%')
            st.progress(result["fake_probability"])

        st.divider()
        st.caption("Model: Fine-tuned DistilBERT")
        st.info(
            "Important: This is a text classifier, not an independent fact-checking "
            "service. Predictions depend on the training data."
        )
