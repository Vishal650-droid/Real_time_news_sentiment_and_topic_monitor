import sys
import os
project_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from src.sentiment_model import get_sentiment_scores

# load dataset
df = pd.read_csv("data/Tweets.csv")

# keep only needed columns
df = df[["text", "airline_sentiment"]]

# rename labels
mapping = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral"
}

df["actual"] = df["airline_sentiment"].map(mapping)

predictions = []

for text in df["text"]:
    try:
        sentiment = get_sentiment_scores(text)
        predictions.append(sentiment["label"])
    except:
        predictions.append("Neutral")

print("Accuracy:")
print(accuracy_score(df["actual"], predictions))

print("\nClassification Report:")
print(classification_report(df["actual"], predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(df["actual"], predictions))


import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(df["actual"], predictions)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Negative", "Neutral", "Positive"],
    yticklabels=["Negative", "Neutral", "Positive"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("VADER Sentiment Evaluation")

plt.savefig("evaluation/confusion_matrix.png")
plt.show()