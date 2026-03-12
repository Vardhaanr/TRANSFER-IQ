import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("Week 3: Advanced Performance Trends")

# LOADING DATASET
df = pd.read_csv("final_modeling_dataset.csv")

st.subheader("Dataset Preview")
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

# TOP PERFORMERS BY FORM TREND
st.subheader("Top 10 Players by Form Trend")
top_form = df.nlargest(10, 'form_trend')[['player', 'form_trend', 'games_played', 'minutes_played']]
st.write(top_form)

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.barh(top_form['player'].iloc[::-1], top_form['form_trend'].iloc[::-1], color='steelblue')
ax1.set_xlabel("Form Trend Score")
ax1.set_title("Top 10 Players by Form Trend")
st.pyplot(fig1)

# GAMES PLAYED DISTRIBUTION
st.subheader("Games Played Distribution")
fig2, ax2 = plt.subplots()
ax2.hist(df['games_played'].dropna(), bins=20, color='coral', edgecolor='black')
ax2.set_xlabel("Games Played")
ax2.set_ylabel("Number of Players")
ax2.set_title("Games Played Distribution")
st.pyplot(fig2)

# PASS ACCURACY ANALYSIS
st.subheader("Pass Accuracy Statistics")
fig3, ax3 = plt.subplots()
ax3.hist(df['pass_accuracy'].dropna(), bins=20, color='lightgreen', edgecolor='black')
ax3.set_xlabel("Pass Accuracy (%)")
ax3.set_ylabel("Number of Players")
ax3.set_title("Pass Accuracy Distribution")
st.pyplot(fig3)

# GOALS VS ASSISTS
st.subheader("Goals vs Assists Scatter Plot")
fig4, ax4 = plt.subplots()
ax4.scatter(df['goals'], df['assists'], alpha=0.6, s=100, color='purple')
ax4.set_xlabel("Goals")
ax4.set_ylabel("Assists")
ax4.set_title("Goals vs Assists")
st.pyplot(fig4)

# TACKLES DISTRIBUTION
st.subheader("Tackles Distribution")
fig5, ax5 = plt.subplots()
ax5.hist(df['tackles'].dropna(), bins=20, color='gold', edgecolor='black')
ax5.set_xlabel("Tackles")
ax5.set_ylabel("Number of Players")
ax5.set_title("Tackles Distribution")
st.pyplot(fig5)

# KEY STATISTICS
st.subheader("Key Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Players", len(df))
with col2:
    st.metric("Avg Form Trend", f"{df['form_trend'].mean():.2f}")
with col3:
    st.metric("Avg Goals", f"{df['goals'].mean():.2f}")

st.write(f"- Avg Games Played: {df['games_played'].mean():.1f}")
st.write(f"- Avg Pass Accuracy: {df['pass_accuracy'].mean():.2f}%")
st.write(f"- Avg Assists: {df['assists'].mean():.2f}")
st.write(f"- Avg Tackles: {df['tackles'].mean():.2f}")
