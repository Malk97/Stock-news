from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
import pandas as pd
import requests
import praw
import ast
from huggingface_hub import InferenceClient


# Load environment variables
NEWSAPI_KEY = "b177a22d92dd4d57952c7ea6d6afee29"
model_path = "Model"
CLIENT_ID = "yu7KW2fhRmQBUW3LDL1X2A"
CLIENT_SECRET = "VZa78pB0H3bVlwOWsnsgV3yzFBLkSw"
REDIRECT_URI = "http://localhost:8000"


client = InferenceClient(
    provider="hf-inference",
    api_key="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",
)


# Define source credibility dictionary
SOURCE_CREDIBILITY_DICT = {
    'cnn': 5, 'bbc': 5, 'fox news': 4, 'al jazeera': 5, 'the new york times': 5,
    'the guardian': 5, 'reuters': 5, 'associated press (ap)': 5, 'washington post': 5,
    'the wall street journal': 5, 'bloomberg': 5, 'npr': 5, 'usa today': 4, 'time': 4,
    'huffpost': 4, 'the independent': 4, 'the telegraph': 4, 'independent.co.uk': 4,
    'sky news': 4, 'abc news': 5, 'nbc news': 5, 'the economist': 5, 'politico': 5,
    'the times': 5, 'new york post': 3, 'daily mail': 3, 'huffpost': 4, 'rt': 2, 'infowars': 1,
    'the sun': 3, 'breitbart': 2, 'the blaze': 2, 'newsmax': 2, 'world news daily report': 1,
    'pravda': 1, 'sputnik': 2, 'the national enquirer': 1, 'business insider': 4,
    'yahoo entertainment': 3, 'wired': 4, 'android central': 4, 'macrumors': 4,
    'marketwatch': 4, 'the verge': 4
}

def fetch_financial_news(num_articles=5):
    """Fetch financial news related to stocks, finance, or markets from News API."""
    url = f"https://newsapi.org/v2/everything?q=stocks OR finance OR markets&apiKey={NEWSAPI_KEY}&language=en&pageSize={num_articles}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        articles = response.json().get("articles", [])
        return pd.DataFrame(articles)
        
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching news: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    
def predict_class(text: str,client):
    """
    Preprocess the text, tokenize it, and predict the class using the model.
    """
    # Preprocess text (this will be handled by the preprocess_text function from text_processing.py)
    text = preprocess_text(text)
    result = client.text_classification(inputs=text ,model="ProsusAI/finbert")
    return result

def process_news_data(df):
    """Process news DataFrame by adding sentiment and source credibility."""
    df = df[['title', 'description', 'url', 'publishedAt', 'source']]
    df['sentiment'] = df['description'].apply(lambda x: predict_class(x,client))
    df['source'] = df['source'].apply(str).apply(ast.literal_eval).apply(lambda x: x.get('name', 'unknown')).str.lower()
    df['source_credibility'] = df['source'].map(SOURCE_CREDIBILITY_DICT).fillna(1)
    return df

def get_ranking_column(df):
    """Add rank column based on sentiment and source credibility."""
    df['Rank'] = df.apply(lambda row: ranking(row['title'], row['sentiment'], row['source_credibility']), axis=1)
    return df

def get_financial_news_with_ranking(num_articles=20):
    """Main function to get, process, and return financial news with ranking."""
    financial_news_df = fetch_financial_news(num_articles)

    if financial_news_df.empty:
        return financial_news_df

    financial_news_df = process_news_data(financial_news_df)
    financial_news_df = get_ranking_column(financial_news_df)
    df_sorted = financial_news_df.sort_values(by='Rank', ascending=False)
    return df_sorted[['title', 'description', 'url', 'source']]



def preprocess_text(text):
    """Dummy function for text preprocessing."""
    # Placeholder function for preprocessing, modify as needed
    return text.lower()

def get_reddit_posts(query):
    reddit = praw.Reddit(client_id=CLIENT_ID,
                         client_secret=CLIENT_SECRET,
                         user_agent=CLIENT_SECRET, 
                         redirect_uri=REDIRECT_URI)

    subreddit = reddit.subreddit('stocks')  
    posts = subreddit.search(query, limit=50, time_filter='week')
    
    post_data = []
    number_of_post = 0
    total = 0

    for submission in posts:
        number_of_post += 1
        score_plus_comments = submission.score + submission.num_comments
        post_data.append(score_plus_comments)
        total += score_plus_comments

    ratio = number_of_post / total if total != 0 else 0  
    
    normalized_values = []
    if post_data:  
        min_value = min(post_data)
        max_value = max(post_data)

        for value in post_data:
            if max_value != min_value:
                normalized_value = (value - min_value) / (max_value - min_value)
            else:
                normalized_value = 0  
            normalized_values.append(normalized_value)
        
    return sum(normalized_values) / number_of_post if number_of_post > 0 else 0

def ranking(text, sentiment_score, source_credibility):
    alpha, beta, gamma = 0.5, 0.3, 0.2
    rank_score = beta * get_reddit_posts(text) + gamma * source_credibility
    return rank_score

# Streamlit sidebar content
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/newspaper-concept-illustration_114360-24706.jpg?t=st=1741964230~exp=1741967830~hmac=667dc595f94d38b742f8f0c37b73e125728113502d799f7253c536c7b37e9f03&w=826", use_column_width=True) 
    st.markdown("<h2 style='font-size: 28px;'>Welcome to Latest News!</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>This platform aggregates the latest financial news articles for you. Stay updated on the stock market.</p>", unsafe_allow_html=True)
    st.markdown('-------------------------------------------------')
    st.markdown("<h4 style='font-size: 18px;'>Select number of articles to display:</h4>", unsafe_allow_html=True)
    num_articles = st.slider("", min_value=5, max_value=50, value=20, step=5) 
    st.write(f"Displaying {num_articles} articles.")

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
