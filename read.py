import json
import pandas as pd

file_path = "open-data-master/data/events/7444.json"

with open(file_path) as f:
    data = json.load(f)

df = pd.json_normalize(data)

df.to_csv("performance_sample.csv", index=False)

goals = df[df['shot.outcome.name'] == 'Goal']
print("GOALS:")
print(goals['player.name'].value_counts().head())

assists = df[df['pass.goal_assist'] == True]
print("\nASSISTS:")
print(assists['player.name'].value_counts().head())

passes = df[df['type.name'] == 'Pass']
print("\nPASSES:")
print(passes['player.name'].value_counts().head())

tackles = df[df['type.name'] == 'Duel']
print("\nTACKLES:")
print(tackles['player.name'].value_counts().head())

df['time_seconds'] = df['minute'].fillna(0) * 60 + df['second'].fillna(0)
minutes_played = df.groupby('player.name')['time_seconds'].max() / 60

print("\nMINUTES PLAYED:")
print(minutes_played.sort_values(ascending=False).head())