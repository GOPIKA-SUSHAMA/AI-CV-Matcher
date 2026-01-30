import streamlit as st
import re
from sentence_transformers import SentenceTransformer, util

# Load NLP Model

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ----------------------------
# Text Preprocessing
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.strip()

# ----------------------------
# Core NLP Similarity Function
# ----------------------------
def jd_cv_similarity(cv_text, jd_text):
    cv_clean = clean_text(cv_text)
    jd_clean = clean_text(jd_text)

    cv_emb = model.encode(cv_clean, convert_to_tensor=True)
    jd_emb = model.encode(jd_clean, convert_to_tensor=True)

    score = util.cos_sim(cv_emb, jd_emb)
    return float(score)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI CV–JD Matcher", layout="centered")

st.title("AI Job–Candidate Matcher (NLP)")
st.write("Semantic matching of CV and Job Description using NLP.")

cv_text = st.text_area(" Paste Candidate CV Text", height=220)
jd_text = st.text_area(" Paste Job Description Text", height=220)

if st.button(" Calculate Match Score"):
    if cv_text.strip() and jd_text.strip():
        with st.spinner("Analyzing model..."):
            score = jd_cv_similarity(cv_text, jd_text)

        st.metric("Semantic Match Score", f"{score:.2f}")

        if score >= 0.75:
            st.success("Strong Match ")
        elif score >= 0.50:
            st.warning("Moderate Match ")
        else:
            st.error("Weak Match ")

        st.caption("Score based on Sentence-BERT semantic similarity.")
    else:
        st.info("Please paste both CV and Job Description.")
