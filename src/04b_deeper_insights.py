import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from textblob import TextBlob

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = "E:\\3rd_year\\6SEM\\Project\\Disaster_NLP_Project\\data\\processed"
VISUALS_PATH = "E:\3rd_year\6SEM\Project\Disaster_NLP_Project\visuals"
os.makedirs(VISUALS_PATH, exist_ok=True)

# --- 1. LOAD DATA ---
print("Loading data for deeper insights...")
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv"))
df = df.dropna(subset=['cleaned_text'])

# --- FEATURE EXTRACTION (Meta-Features) ---
print("Extracting NLP Meta-Features (Sentiment, Char Count, Punctuation)...")
# 1. Sentiment Polarity (-1.0 is highly negative/sad, +1.0 is highly positive/happy)
df['sentiment'] = df['cleaned_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# 2. Character Count
df['char_count'] = df['cleaned_text'].apply(lambda x: len(str(x)))

# 3. Word Count
df['word_count'] = df['cleaned_text'].apply(lambda x: len(str(x).split()))

# --- INSIGHT 5: SENTIMENT DENSITY PLOT (KDE) ---
print("Generating Sentiment Density Plot...")
plt.figure(figsize=(10, 6))
# Plot the distribution of sentiment for each urgency level
sns.kdeplot(data=df[df['urgency'] == 'High']['sentiment'], label='High Urgency', fill=True, color='#d62728', alpha=0.5)
sns.kdeplot(data=df[df['urgency'] == 'Low']['sentiment'], label='Low Urgency', fill=True, color='#2ca02c', alpha=0.5)

plt.title("Sentiment Polarity Distribution: High vs Low Urgency", fontsize=14, fontweight='bold')
plt.xlabel("Sentiment Score (-1.0 = Negative/Panic, 1.0 = Positive/Relief)", fontsize=12)
plt.ylabel("Density of Tweets", fontsize=12)
plt.legend()
plt.axvline(x=0, color='black', linestyle='--', alpha=0.7) # Add a line at neutral (0)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "sentiment_density_kde.png"), dpi=300)

# --- INSIGHT 6: VIOLIN PLOT FOR CHARACTER COUNT ---
print("Generating Character Count Violin Plot...")
plt.figure(figsize=(9, 6))
sns.violinplot(x='urgency', y='char_count', data=df, order=['Low', 'Medium', 'High'], palette=['#2ca02c', '#ff7f0e', '#d62728'], inner="quartile")
plt.title("Violin Plot: Density of Tweet Character Lengths", fontsize=14, fontweight='bold')
plt.xlabel("Urgency Level", fontsize=12)
plt.ylabel("Character Count per Tweet", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "tweet_length_violin.png"), dpi=300)

# --- INSIGHT 7: META-FEATURE CORRELATION HEATMAP ---
print("Generating Feature Correlation Heatmap...")
# Select only our numerical features to see how they interact
numerical_features = df[['sentiment', 'char_count', 'word_count']]
correlation_matrix = numerical_features.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, linecolor='black')
plt.title("Correlation Heatmap of Tweet Meta-Features", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "meta_feature_heatmap.png"), dpi=300)

print("\n🚀 Deeper visual insights complete! Check the 'visuals' folder.")