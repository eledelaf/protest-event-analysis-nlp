"""
This script plots the evolution of topics over time.
I want each topic to have a different color, create a plot that shows each topic vs time, 
so I can see when are we talking more about each one.
"""
import csv
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# 1. Load your CSV file
df = pd.read_csv('6.Topic_analysis/topic_modeling/articles_with_topics.csv', sep=',', encoding='utf-8')

# 2. Select the desired columns
desired_columns = ['url','published_date', 'topic']
df_clean = df[desired_columns]
df_clean['published_date'] = pd.to_datetime(df_clean['published_date'])

# drop rows where published_date is null
df_clean = df_clean.dropna(subset=['published_date'])


# Grouping topics for better visualization
topic_groups = {
    "BLM": [-1, 2],
    "COVID": [4, 6],
    "Gaza": [5],
    "Ukraine": [7],
    "Capitol": [0],
    "Anti-immigration": [1]
}

# Add the topic group names to the data
df_clean['topic_group'] = df_clean['topic'].map(lambda x: next((k for k, v in topic_groups.items() if x in v), "Other"))

# Filter by topic group 
df_clean = df_clean[df_clean['topic_group'].isin(topic_groups.keys())]

# 3. Make month-year bins
df_clean['month_year'] = df_clean['published_date'].dt.to_period('M').dt.to_timestamp()

# 3.1 Count articles per topic and month
topic_month_counts = df_clean.groupby(['month_year', 'topic_group']).size().reset_index(name='count')

# 4. Plot the evolution of topics over time
plt.figure(figsize=(12, 6))

for group_name in topic_groups.keys():   # group_name = "COVID", "BLM", ...
    data = topic_month_counts[topic_month_counts["topic_group"] == group_name]
    plt.plot(data["month_year"], data["count"], label=group_name)

plt.xlabel("Month-Year")
plt.ylabel("Number of Articles")
plt.title("Evolution of Topic Groups Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("7.2figures/topic_by_time.png")
plt.show()

# Find the peak of each topic 
peak_topics = topic_month_counts.loc[topic_month_counts.groupby('topic_group')['count'].idxmax()]
print(peak_topics)