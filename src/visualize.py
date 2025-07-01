from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

def show_wordcloud(keywords):
    text = ' '.join(keywords)
    wc = WordCloud(width=600, height=300, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def show_sentiment_chart(sentiments, key=None):
    fig = go.Figure(go.Bar(
        x=['Sentiment'],
        y=[sum(sentiments)/len(sentiments) if sentiments else 0],
        marker_color='green' if sum(sentiments) >= 0 else 'red'
    ))
    fig.update_layout(title='Average Sentiment', yaxis=dict(range=[-1, 1]))
    st.plotly_chart(fig, key=key)

