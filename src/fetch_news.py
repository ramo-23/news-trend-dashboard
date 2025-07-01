import requests
import os

def fetch_news(city, api_key=None):
    if api_key is None:
        api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("API key not found. Set the NEWSAPI_KEY environment variable.")
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={city}&"
        f"sortBy=publishedAt&"
        f"language=en&"
        f"apiKey={api_key}"
    )
    response = requests.get(url)
    data = response.json()
    return data.get("articles", [])

