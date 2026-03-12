import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("Week 4: Sentiment Analysis & Final Integration")

# LOADING DATASET
df = pd.read_csv("final_modeling_dataset.csv")

st.subheader("Final Merged Dataset Preview")
st.write(df.head())

# PLAYER SEARCH
st.subheader("Player Search")
player_name = st.text_input("Enter Player Name")

if player_name:
    player_data = df[df["player"].str.contains(player_name, case=False, na=False)]
    if not player_data.empty:
        st.write(player_data)
    else:
        st.write("Player not found")

# SENTIMENT DISTRIBUTION
st.subheader("Sentiment Scores Distribution")

# Extract valid sentiment data
sentiment_data = df[['sentiment_positive', 'sentiment_negative', 'sentiment_compound']].dropna()

if len(sentiment_data) > 0:
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    ax1.hist(sentiment_data['sentiment_positive'], bins=10, color='green', alpha=0.7, edgecolor='black')
    ax1.set_xlabel("Positive Sentiment")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Positive Sentiment Distribution")
    
    ax2.hist(sentiment_data['sentiment_negative'], bins=10, color='red', alpha=0.7, edgecolor='black')
    ax2.set_xlabel("Negative Sentiment")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Negative Sentiment Distribution")
    
    ax3.hist(sentiment_data['sentiment_compound'], bins=10, color='blue', alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Compound Sentiment")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Compound Sentiment Distribution")
    
    st.pyplot(fig1)
else:
    st.write("Limited sentiment data available")

# MARKET VALUE VS FORM TREND
st.subheader("Market Value vs Form Trend")
df_clean = df.dropna(subset=['market_value', 'form_trend'])
if len(df_clean) > 0:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    scatter = ax2.scatter(df_clean['form_trend'], df_clean['market_value'], 
                         alpha=0.6, s=100, c=df_clean['goals'], cmap='viridis')
    ax2.set_xlabel("Form Trend")
    ax2.set_ylabel("Market Value")
    ax2.set_title("Market Value vs Form Trend (colored by Goals)")
    plt.colorbar(scatter, ax=ax2, label="Goals")
    st.pyplot(fig2)

# TOP PLAYERS BY GOALS
st.subheader("Top 10 Players by Goals")
top_goals = df.nlargest(10, 'goals')[['player', 'goals', 'assists', 'form_trend', 'market_value']]
st.write(top_goals)

# TOP PERFORMERS BY FORM TREND
st.subheader("Top 10 Players by Form Trend with Sentiment")
top_form = df.nlargest(10, 'form_trend')[['player', 'form_trend', 'goals', 'assists', 
                                           'sentiment_positive', 'sentiment_negative', 'sentiment_compound']]
st.write(top_form)

# PASSES VS GOALS CORRELATION
st.subheader("Total Passes vs Goals")
fig3, ax3 = plt.subplots(figsize=(10, 5))
df_passes = df.dropna(subset=['total_passes', 'goals'])
ax3.scatter(df_passes['total_passes'], df_passes['goals'], alpha=0.6, s=100, color='orange')
ax3.set_xlabel("Total Passes")
ax3.set_ylabel("Goals")
ax3.set_title("Total Passes vs Goals Scored")
st.pyplot(fig3)

# KEY METRICS & CORRELATIONS
st.subheader("Dataset Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Players", len(df))
with col2:
    st.metric("Total Features", len(df.columns))
with col3:
    st.metric("Avg Goals", f"{df['goals'].mean():.2f}")
with col4:
    st.metric("Avg Assists", f"{df['assists'].mean():.2f}")

st.write(f"- Total Games Played: {df['games_played'].sum():.0f}")
st.write(f"- Avg Minutes Played: {df['minutes_played'].mean():.1f}")
st.write(f"- Avg Pass Accuracy: {df['pass_accuracy'].mean():.2f}%")
st.write(f"- Total Tackles: {df['tackles'].sum():.0f}")
