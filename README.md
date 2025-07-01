# 🌐 AI-Powered Local News Trends Dashboard

**Discover the pulse of the world’s cities in real time!**  
This dashboard uses the latest in NLP, clustering, and interactive visualization to reveal what’s trending in your city—powered by AI and open data.

---

## 🚀 What Makes This Project Awesome?

- **City Explorer:** Instantly analyze news trends for thousands of cities worldwide.
- **Live News:** Fetches real-time articles using [NewsAPI](https://newsapi.org/).
- **NLP Magic:** Extracts keywords, analyzes sentiment, and summarizes topics using advanced models like **BERTopic** and **TextBlob**.
- **Clustering:** Groups articles into topics with BERTopic for deep trend insights.
- **Visualizations:** Interactive word clouds, sentiment charts, and keyword timelines with **Plotly** and **WordCloud**.
- **Compare Cities:** See how news and sentiment stack up between any two cities.
- **Sentiment Map:** Explore a global map of news sentiment for capital cities.
- **Favorites:** Save your favorite cities and articles for quick access.

---

## 🛠️ Tech Stack

- **Streamlit** for the interactive dashboard UI
- **NewsAPI** for fetching live news
- **NLTK** and **TextBlob** for keyword extraction and sentiment analysis
- **BERTopic** and **sentence-transformers** for topic modeling and clustering
- **Plotly** and **WordCloud** for beautiful, interactive charts
- **Pandas** for data wrangling
- **Python-dotenv** for secure API key management

---

## 📦 Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/yourusername/news-trend-dashboard.git
cd news-trend-dashboard
```

### 2. Download the World Cities Database

- **Required:** Download the free [world cities CSV](https://simplemaps.com/data/world-cities) from SimpleMaps.
- Place `worldcities.csv` in the `data/` folder of this project.

### 3. Set up a virtual environment

```sh
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 4. Install dependencies

```sh
pip install -r requirements.txt
```

### 5. Set up your NewsAPI key

- Create a `.env` file in the project root:
  ```
  NEWSAPI_KEY=your_api_key_here
  ```

### 6. Run the dashboard

```sh
streamlit run app.py
```

---

## 🗂️ Project Structure

```
.
├── app.py
├── requirements.txt
├── .env
├── data/
│   └── worldcities.csv   # <--- Download from SimpleMaps!
├── src/
│   ├── fetch_news.py
│   ├── process_text.py
│   └── visualize.py
├── favorites.json
├── city_sentiments.json
└── README.md
```

---

## ⚡ Notes & Tips

- **Privacy:** Your `.env` and `venv/` folders are gitignored—keep your API keys safe!
- **Performance:** For best results, use Python 3.8 or newer.
- **Customization:** Tweak the sidebar to explore more cities, articles, or trends.
- **Data:** The dashboard works best with the latest `worldcities.csv` from SimpleMaps.

---

## 💡 Why This Project?

This dashboard is built for journalists, researchers, and the curious—anyone who wants to see how news and sentiment flow across the globe, city by city, in real time.  
It’s a showcase of modern NLP, clustering, and data visualization—all in one place.

---

Made with ❤️ by Ramokhele Manyeli  
*Powered by Streamlit, NewsAPI, BERTopic, and Python*

