from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
import pandas as pd
import requests
import praw
import ast
from huggingface_hub import InferenceClient

# Load environment variables
NEWSAPI_KEY = "b177a22d92dd4d57952c7ea6d6afee29"
CLIENT_ID = "yu7KW2fhRmQBUW3LDL1X2A"
CLIENT_SECRET = "VZa78pB0H3bVlwOWsnsgV3yzFBLkSw"
REDIRECT_URI = "http://localhost:8000"

# Initialize Hugging Face Inference Client
client = InferenceClient(
    provider="hf-inference",
    api_key="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",
)

def predict_class(text: str):
    """استخدام API لتصنيف النص إلى إيجابي أو سلبي"""
    result = client.text_classification(
        inputs=text,
        model="ProsusAI/finbert",
    )
    label = max(result, key=lambda x: x['score'])['label']
    return "Positive" if label == "positive" else "Negative"

# تحديث `process_news_data` لاستخدام `predict_class`
def process_news_data(df):
    """إضافة تصنيف المشاعر ومصداقية المصدر للبيانات"""
    df = df[['title', 'description', 'url', 'publishedAt', 'source']].dropna()
    df['sentiment'] = df['description'].apply(predict_class)
    df['source'] = df['source'].apply(str).apply(ast.literal_eval).apply(lambda x: x.get('name', 'unknown')).str.lower()
    df['source_credibility'] = df['source'].map(SOURCE_CREDIBILITY_DICT).fillna(1)
    return df

# إصلاح `ranking` بحيث يتعامل مع غياب بيانات Reddit
def ranking(text, sentiment_score, source_credibility):
    alpha, beta, gamma = 0.5, 0.3, 0.2
    reddit_score = get_reddit_posts(text)
    rank_score = beta * reddit_score + gamma * source_credibility
    return rank_score

# Fetch and display the data
data = get_financial_news_with_ranking(num_articles)

if isinstance(data, pd.DataFrame) and not data.empty:
    st.title("Latest News")
    for _, article in data.iterrows():
        with st.container():
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.write(f"**Description Preview:** {article['description'][:300]}...")
            st.markdown(f"[Read full article]({article['url']})")
            st.write(f"**Source:** {article['source']}")
            st.markdown("---")
else:
    st.write("There was an error loading the data.")
