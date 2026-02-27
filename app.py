import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter

# Optional imports with safe fallback
try:
    from textblob import TextBlob
except:
    TextBlob = None

try:
    from keybert import KeyBERT
    kw_model = KeyBERT()
except:
    kw_model = None


# =========================
# LLM PLACEHOLDER FUNCTION
# =========================
def llm_generate(prompt):
    """
    Replace this function with OpenAI / Ollama API integration
    """
    return f"LLM Output Preview:\n\n{prompt[:300]}..."


# =========================
# TEXT PREPROCESSING
# =========================
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =========================
# LLM FEATURES
# =========================
def summarize(text):
    return llm_generate(f"Summarize this news article:\n{text}")

def simple_explain(text):
    return llm_generate(f"Explain in simple language:\n{text}")

def explain_like_10(text):
    return llm_generate(f"Explain like I am 10 years old:\n{text}")

def linkedin_post(text):
    return llm_generate(f"Convert into professional LinkedIn post:\n{text}")

def key_insights(text):
    return llm_generate(f"Generate key insights from article:\n{text}")


# =========================
# SENTIMENT ANALYSIS
# =========================
def sentiment_analysis(text):
    if TextBlob is None:
        return "TextBlob not installed"

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive 😊"
    elif polarity < 0:
        return "Negative 😞"
    else:
        return "Neutral 😐"


# =========================
# KEYWORD EXTRACTION
# =========================
def extract_keywords(text):
    if kw_model is None:
        return ["KeyBERT not installed"]

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=10
    )
    return [k[0] for k in keywords]


# =========================
# TREND GRAPH
# =========================
def keyword_trend(text):
    words = re.findall(r'\w+', text.lower())
    count = Counter(words)
    common = count.most_common(10)

    if not common:
        return None

    df = pd.DataFrame(common, columns=["Keyword", "Frequency"])

    fig, ax = plt.subplots()
    ax.bar(df["Keyword"], df["Frequency"])
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="AI News Research App", layout="wide")

st.title("🧠 AI News Research & Smart Summarizer")

st.markdown("Paste any news article and get AI-powered insights instantly.")

news_input = st.text_area("📄 Paste News Article Here", height=250)

if st.button("🚀 Analyze"):

    if not news_input.strip():
        st.warning("Please paste a news article.")
        st.stop()

    text = preprocess_text(news_input)

    with st.spinner("Analyzing article..."):

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Summary")
            st.write(summarize(text))

            st.subheader("😊 Sentiment")
            st.write(sentiment_analysis(text))

            st.subheader("🔑 Keywords")
            keywords = extract_keywords(text)
            st.write(keywords)

        with col2:
            st.subheader("💡 Key Insights")
            st.write(key_insights(text))

            st.subheader("💼 LinkedIn Post")
            st.write(linkedin_post(text))

        st.subheader("🪄 Explain Simply")
        st.write(simple_explain(text))

        st.subheader("🧒 Explain Like I am 10")
        st.write(explain_like_10(text))

        st.subheader("📊 Keyword Trend Graph")
        fig = keyword_trend(text)
        if fig:
            st.pyplot(fig)
        else:
            st.info("Not enough data to generate graph.")
