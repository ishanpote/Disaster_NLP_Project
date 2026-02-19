import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = "../data/processed/"
VISUALS_PATH = "../visuals/"
os.makedirs(VISUALS_PATH, exist_ok=True)

# Load Data
print("Loading data for advanced insights...")
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv"))
df = df.dropna(subset=['cleaned_text'])

# --- INSIGHT 1: ORIGINAL CLASS BREAKDOWN ---
print("1. Generating Original Class Breakdown...")
plt.figure(figsize=(10, 6))
class_counts = df['class_label'].value_counts()
sns.barplot(x=class_counts.values, y=class_counts.index, palette="viridis")
plt.title("Distribution of Original Disaster Categories", fontweight='bold')
plt.xlabel("Number of Tweets")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "class_breakdown.png"), dpi=300)

# --- INSIGHT 2: TWEET LENGTH BY URGENCY ---
print("2. Generating Tweet Length Analysis...")
# Calculate word count for each tweet
df['word_count'] = df['cleaned_text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(8, 6))
sns.boxplot(x='urgency', y='word_count', data=df, order=['Low', 'Medium', 'High'], palette=['#2ca02c', '#ff7f0e', '#d62728'])
plt.title("Tweet Length (Word Count) by Urgency Level", fontweight='bold')
plt.xlabel("Urgency Level")
plt.ylabel("Number of Words per Tweet")
plt.ylim(0, 50) # Cap at 50 words for better visibility
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "tweet_length_boxplot.png"), dpi=300)

# --- INSIGHT 3: HIGH-URGENCY BIGRAMS (2-Word Phrases) ---
print("3. Generating Top Bigrams for High Urgency...")
high_urgency_df = df[df['urgency'] == 'High']['cleaned_text']

# Extract 2-word combinations
vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english', max_features=15)
bigrams = vectorizer.fit_transform(high_urgency_df)
bigram_counts = pd.DataFrame(bigrams.sum(axis=0).T, index=vectorizer.get_feature_names_out(), columns=['Freq'])
bigram_counts = bigram_counts.sort_values(by='Freq', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=bigram_counts['Freq'], y=bigram_counts.index, palette="Reds_r")
plt.title("Top 15 Bigrams (2-Word Phrases) in HIGH Urgency Tweets", fontweight='bold')
plt.xlabel("Frequency")
plt.ylabel("Phrase")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "high_urgency_bigrams.png"), dpi=300)

# --- INSIGHT 4: WORD CLOUDS (High vs Low) ---
print("4. Generating Word Clouds...")
def generate_wordcloud(text_data, filename, colormap):
    text = " ".join(text_data)
    wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap, max_words=100).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_PATH, filename), dpi=300)
    plt.close()

# Generate for High Urgency (Red theme)
generate_wordcloud(df[df['urgency'] == 'High']['cleaned_text'], "wordcloud_high_urgency.png", 'Reds')
# Generate for Low Urgency (Green theme)
generate_wordcloud(df[df['urgency'] == 'Low']['cleaned_text'], "wordcloud_low_urgency.png", 'Greens')

print("🚀 All advanced visualizations saved in the 'visuals/' folder!")