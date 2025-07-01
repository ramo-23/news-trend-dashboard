import streamlit as st
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
from bertopic import BERTopic
from collections import defaultdict, Counter
import os
import json

# Import custom modules for news fetching, NLP, and visualization
from src.fetch_news import fetch_news
from src.process_text import extract_keywords, analyze_sentiment, cluster_articles, bertopic_clusters, summarize_topic
from src.visualize import show_wordcloud, show_sentiment_chart

# Load environment variables (e.g., API keys)
load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(
    page_title="AI-Powered Local News Trends Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load city data from CSV and filter for cities with population > 100,000
@st.cache_data
def load_cities():
    df = pd.read_csv(os.path.join("data", "worldcities.csv"))
    df = df[df['population'] > 100000]
    return df

cities_df = load_cities()
primary_capitals_df = cities_df[cities_df['capital'] == 'primary']

# Sidebar navigation
st.sidebar.title("🗺️ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Articles", "Visualizations", "Trends", "Clusters", "Compare Cities", "Sentiment Map", "Favorites"],
    index=0
)

# Sidebar: City selection and settings
st.sidebar.title("🌍 Settings")
city = st.sidebar.selectbox(
    "Choose your city:",
    options=sorted(cities_df['city_ascii'].unique()),
    index=list(cities_df['city_ascii']).index("Pretoria") if "Pretoria" in list(cities_df['city_ascii']) else 0
)
num_articles = st.sidebar.slider("Number of articles to analyze", 5, 30, 10)

# Sidebar: Compare with another city
st.sidebar.title("🏙️ Compare Cities")
city2 = st.sidebar.selectbox(
    "Compare with another city (optional):",
    options=sorted(cities_df['city_ascii'].unique()),
    index=list(cities_df['city_ascii']).index("New York") if "New York" in list(cities_df['city_ascii']) else 0,
    key="city2"
)
num_articles2 = st.sidebar.slider("Articles for comparison city", 5, 30, 10)

# Display a news article card with keywords and sentiment
def article_card(article, keywords, sentiment):
    sentiment_color = "green" if sentiment > 0.1 else "red" if sentiment < -0.1 else "gray"
    st.markdown(f"#### [{article['title']}]({article['url']})")
    if article.get('urlToImage'):
        st.image(article['urlToImage'], width=500)
    if article.get('description'):
        st.write(article['description'])
    st.write(f"**Keywords:** {', '.join(keywords)}")
    st.markdown(
        f"<span style='color:{sentiment_color}; font-weight:bold;'>Sentiment: {sentiment:.2f}</span>",
        unsafe_allow_html=True
    )
    st.markdown("---")

# Fetch and process news data for a city
@st.cache_data(show_spinner=False)
def get_city_data(city, num_articles):
    articles = fetch_news(city)
    all_keywords, sentiments, dates, sentiment_dates = [], [], [], []
    for idx, article in enumerate(articles[:num_articles]):
        text = article['title'] + '. ' + (article.get('description') or '')
        keywords = extract_keywords(text)
        sentiment = analyze_sentiment(text)
        all_keywords.extend(keywords)
        sentiments.append(sentiment)
        if article.get('publishedAt'):
            date = article['publishedAt'][:10]
            dates.append(date)
            sentiment_dates.append((date, sentiment))
    return articles, all_keywords, sentiments, dates, sentiment_dates

# Format topic names for display
def format_topic_name(name):
    if "_" in name:
        parts = name.split("_")[1:]
        return " ".join([w.capitalize() for w in parts if w])
    return name.capitalize()

# Get top words for a BERTopic topic
def get_top_words(topic_model, topic_num, n_words=5):
    words = topic_model.get_topic(topic_num)
    if words:
        return ", ".join([w for w, _ in words[:n_words]])
    return "N/A"

# Analyze keyword trends over time in articles
def get_keyword_trends(articles, top_n=5):
    date_keywords = defaultdict(list)
    for article in articles:
        date = article.get('publishedAt', '')[:10]
        text = article['title'] + '. ' + (article.get('description') or '')
        kws = extract_keywords(text, num_keywords=top_n)
        date_keywords[date].extend(kws)
    all_keywords = [kw for kws in date_keywords.values() for kw in kws]
    top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(top_n)]
    trend_data = []
    for date in sorted(date_keywords.keys()):
        row = {'date': date}
        kws = date_keywords[date]
        for kw in top_keywords:
            row[kw] = kws.count(kw)
        trend_data.append(row)
    df = pd.DataFrame(trend_data)
    df = df.fillna(0)
    return df, top_keywords

# Get the most representative article title for a topic cluster
def get_representative_title(articles, indices):
    if not indices:
        return "No articles"
    best_idx = max(indices, key=lambda i: len((articles[i].get('description') or '') + articles[i]['title']))
    return articles[best_idx]['title']

# Initialize session state for favorites if not present
if "favorite_cities" not in st.session_state:
    st.session_state.favorite_cities = []
if "favorite_articles" not in st.session_state:
    st.session_state.favorite_articles = []

FAV_FILE = "favorites.json"

# Load favorites from disk
def load_favorites():
    if os.path.exists(FAV_FILE):
        with open(FAV_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"cities": [], "articles": []}

# Save favorites to disk
def save_favorites(favorites):
    with open(FAV_FILE, "w", encoding="utf-8") as f:
        json.dump(favorites, f, ensure_ascii=False, indent=2)

# Load favorites into session state
if "favorites" not in st.session_state:
    st.session_state.favorites = load_favorites()

# Sidebar: Add/remove favorite cities
if city not in st.session_state.favorites["cities"]:
    if st.sidebar.button(f"⭐ Save {city} as favorite"):
        st.session_state.favorites["cities"].append(city)
        save_favorites(st.session_state.favorites)
else:
    st.sidebar.write(f"⭐ {city} is in your favorites")

# Sidebar: Display favorite cities
if st.session_state.favorites["cities"]:
    st.sidebar.markdown("**Your Favorite Cities:**")
    for fav_city in st.session_state.favorites["cities"]:
        st.sidebar.write(f"• {fav_city}")

SENTIMENT_CACHE_FILE = "city_sentiments.json"

# Load cached city sentiment data
def load_city_sentiments():
    if os.path.exists(SENTIMENT_CACHE_FILE):
        with open(SENTIMENT_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Save city sentiment data to disk
def save_city_sentiments(cache):
    with open(SENTIMENT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# Main page header and description
st.markdown(
    "<h1 style='font-size:3em;'>AI-Powered Local News Trends Dashboard</h1>"
    "<p style='font-size:1.2em;'>Discover what's trending in your city, powered by real-time news, NLP, clustering, and beautiful visualizations.</p>",
    unsafe_allow_html=True
)

# Main navigation logic for each page
if page == "Overview":
    st.header(f"📍 {city} - Overview")
    st.write("Use the sidebar to explore articles, visualizations, clusters, or compare cities.")

elif page == "Articles":
    st.header(f"📰 Top Articles in {city}")
    articles, all_keywords, sentiments, dates, sentiment_dates = get_city_data(city, num_articles)
    if not articles:
        st.warning(f"No articles found for {city}. Try another city.")
    else:
        for idx, article in enumerate(articles[:num_articles]):
            text = article['title'] + '. ' + (article.get('description') or '')
            keywords = extract_keywords(text)
            sentiment = analyze_sentiment(text)
            with st.expander(f"{idx+1}. {article['title']}"):
                article_card(article, keywords, sentiment)
                # Favorite articles logic
                if article['url'] not in [a['url'] for a in st.session_state.favorites["articles"]]:
                    if st.button(f"⭐ Save to favorites", key=f"fav_{idx}"):
                        st.session_state.favorites["articles"].append(article)
                        save_favorites(st.session_state.favorites)
                else:
                    st.write("⭐ Saved to favorites")
        st.subheader("Word Cloud of Keywords")
        if all_keywords:
            show_wordcloud(all_keywords)
        else:
            st.info("No keywords found to generate a word cloud.")
        st.subheader("Average Sentiment")
        if sentiments:
            show_sentiment_chart(sentiments)
        else:
            st.info("No sentiment data available.")

elif page == "Clusters":
    st.header(f"🧩 BERTopic Trend Clusters for {city}")
    articles, _, _, _, _ = get_city_data(city, num_articles)
    min_articles = 3
    if len(articles) >= min_articles:
        n_topics = min(5, len(articles) - 1)
        topics, topic_keywords, topic_articles, topic_model = bertopic_clusters(articles, n_topics=n_topics)
        for topic_num, indices in topic_articles.items():
            if topic_num == -1:
                continue
            rep_idx = max(indices, key=lambda i: len((articles[i].get('description') or '') + articles[i]['title']))
            cluster_title = articles[rep_idx]['title']
            top_words = get_top_words(topic_model, topic_num)
            st.markdown(f"<span style='font-size:1.3em;'>🧩 <b>Topic {topic_num+1}:</b> {cluster_title}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#888;'>Top Keywords:</span> <b>{top_words}</b>", unsafe_allow_html=True)
            topic_arts = [articles[rep_idx]]
            with st.spinner("Generating summary..."):
                summary = summarize_topic(topic_arts)
            st.info(f"**Summary:** {summary}")
            with st.expander(f"Show {min(3, len(indices))} sample articles"):
                for idx in indices[:3]:
                    article = articles[idx]
                    st.write(f"- [{article['title']}]({article['url']})")
    else:
        st.info("Not enough articles for clustering.")

elif page == "Compare Cities":
    st.header(f"🔀 Comparing {city} and {city2}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"📍 {city}")
        articles, all_keywords, sentiments, dates, sentiment_dates = get_city_data(city, num_articles)
        st.write(f"Found {len(articles)} articles.")
        if all_keywords:
            show_wordcloud(all_keywords)
        if sentiments:
            show_sentiment_chart(sentiments)
    with col2:
        st.subheader(f"📍 {city2}")
        articles2, all_keywords2, sentiments2, dates2, sentiment_dates2 = get_city_data(city2, num_articles2)
        st.write(f"Found {len(articles2)} articles.")
        if all_keywords2:
            show_wordcloud(all_keywords2)
        if sentiments2:
            show_sentiment_chart(sentiments2)

elif page == "Trends":
    st.header(f"📈 Trends for {city}")
    articles, all_keywords, sentiments, dates, sentiment_dates = get_city_data(city, num_articles)
    if dates:
        st.subheader("Article Publication Timeline")
        df = pd.DataFrame({'date': dates})
        fig = px.histogram(df, x='date', nbins=len(set(dates)), title="Publication Dates")
        st.plotly_chart(fig, use_container_width=True)
    if sentiment_dates:
        st.subheader("Sentiment Over Time")
        df_sent = pd.DataFrame(sentiment_dates, columns=['date', 'sentiment'])
        fig2 = px.line(df_sent.sort_values('date'), x='date', y='sentiment', markers=True, title="Sentiment Over Time")
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Keyword Trend Timeline")
    df_trend, top_keywords = get_keyword_trends(articles, top_n=5)
    if not df_trend.empty:
        col1, col2 = st.columns([2, 3])
        with col1:
            selected_kw = st.selectbox("Select a top keyword:", top_keywords)
        with col2:
            custom_kw = st.text_input("Or enter a custom keyword:")
        keyword = custom_kw.strip() if custom_kw.strip() else selected_kw
        fig_kw = px.line(df_trend, x='date', y=keyword, markers=True, title=f"Trend for '{keyword}'") if keyword in df_trend.columns else None
        if fig_kw:
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info(f"No trend data for '{keyword}'.")
        st.markdown(f"**Articles mentioning '{keyword}':**")
        found = False
        for article in articles:
            text = article['title'] + '. ' + (article.get('description') or '')
            if keyword in extract_keywords(text, num_keywords=10):
                st.write(f"- [{article['title']}]({article['url']})")
                found = True
        if not found:
            st.info(f"No articles found mentioning '{keyword}'.")
    else:
        st.info("Not enough data for keyword trends.")

elif page == "Sentiment Map":
    st.header("🗺️ Sentiment Map of World Capitals")
    st.info("This map shows average news sentiment for primary capital cities.")
    max_cities = st.sidebar.slider("Number of capitals on map", 5, 100, 50, step=5)
    primary_capitals_df = cities_df[cities_df['capital'] == 'primary']
    cities_subset = primary_capitals_df.sort_values("population", ascending=False).head(max_cities)
    map_data = []
    city_sentiments = load_city_sentiments()
    progress = st.progress(0)
    total = len(cities_subset)
    for i, row in enumerate(cities_subset.itertuples()):
        city_name = row.city_ascii
        lat = row.lat
        lng = row.lng
        country = row.country
        if city_name in city_sentiments:
            avg_sentiment = city_sentiments[city_name]
        else:
            articles, _, sentiments, _, _ = get_city_data(city_name, num_articles=10)
            avg_sentiment = sum(sentiments)/len(sentiments) if sentiments else 0
            city_sentiments[city_name] = avg_sentiment
            if i % 50 == 0:
                save_city_sentiments(city_sentiments)
        map_data.append({
            "city": city_name,
            "country": country,
            "lat": lat,
            "lng": lng,
            "sentiment": avg_sentiment
        })
        progress.progress((i + 1) / total)
    save_city_sentiments(city_sentiments)
    progress.empty()
    if not map_data:
        st.info("No data to display on the map.")
    else:
        df_map = pd.DataFrame(map_data)
        fig = px.scatter_geo(
            df_map,
            lat="lat",
            lon="lng",
            color="sentiment",
            hover_name="city",
            hover_data=["country", "sentiment"],
            color_continuous_scale="RdYlGn",
            projection="natural earth",
            size_max=20,
            title="Average Sentiment by City"
        )
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown(
    "<hr><center><small>Made with ❤️ by Ramokhele Manyeli using Streamlit, NewsAPI, and NLP.<br>"
    "Try different cities!</small></center></hr>", unsafe_allow_html=True)