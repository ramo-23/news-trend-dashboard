import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from bertopic import BERTopic
from transformers import pipeline
import os

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_keywords(text, num_keywords=5):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha() and w not in stop_words]
    freq = nltk.FreqDist(words)
    return [word for word, count in freq.most_common(num_keywords)]

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def cluster_articles(articles, n_clusters=3):
    texts = [a['title'] + '. ' + (a.get('description') or '') for a in articles]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels

def bertopic_clusters(articles, n_topics=5):
    texts = [a['title'] + '. ' + (a.get('description') or '') for a in articles]
    if not texts:
        return [], {}, {}, None
    topic_model = BERTopic(top_n_words=5, nr_topics=n_topics, calculate_probabilities=False, verbose=False)
    topics, _ = topic_model.fit_transform(texts)
    topic_info = topic_model.get_topic_info()
    topic_keywords = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
    topic_articles = {}
    for idx, topic in enumerate(topics):
        topic_articles.setdefault(topic, []).append(idx)
    return topics, topic_keywords, topic_articles, topic_model

def summarize_topic(articles, max_words=100):
    context = " ".join(a['title'] + ". " + (a.get('description') or "") for a in articles)
    try:
        summary = summarizer(context, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Summary unavailable: {e}"

