import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = "E:\\3rd_year\\6SEM\\Project\\Disaster_NLP_Project\\data\\processed"
VISUALS_PATH = "../visuals/"

# Create a folder to save your graphs
os.makedirs(VISUALS_PATH, exist_ok=True)

# --- 1. LOAD DATA ---
print("Loading processed data for visualization...")
try:
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv"))
    df = df.dropna(subset=['cleaned_text'])
    print("✅ Data Loaded.")
except FileNotFoundError:
    print("❌ Error: humaid_cleaned.csv not found.")
    exit()

# --- 2. PLOT 1: URGENCY DISTRIBUTION ---
print("Generating Urgency Distribution Chart...")
plt.figure(figsize=(8, 6))

# Define colors for urgency
colors = ['#2ca02c', '#ff7f0e', '#d62728'] # Green (Low), Orange (Medium), Red (High)
urgency_counts = df['urgency'].value_counts()

# Ensure the order is Low, Medium, High for the chart
urgency_counts = urgency_counts[['Low', 'Medium', 'High']]

urgency_counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Distribution of Urgency Levels in Disaster Tweets', fontsize=14, fontweight='bold')
plt.xlabel('Urgency Level', fontsize=12)
plt.ylabel('Number of Tweets', fontsize=12)
plt.xticks(rotation=0, fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
dist_path = os.path.join(VISUALS_PATH, 'urgency_distribution.png')
plt.tight_layout()
plt.savefig(dist_path, dpi=300) # High resolution for presentations
print(f"✅ Saved chart to: {dist_path}")

# --- 3. PLOT 2: TOP WORDS IN HIGH URGENCY TWEETS ---
print("Generating Top Words Chart for High Urgency...")

# Get only High Urgency text
high_urgency_text = " ".join(df[df['urgency'] == 'High']['cleaned_text'])
words = high_urgency_text.split()

# Remove standard stop words that might have slipped through
stop_words = {'the', 'to', 'and', 'in', 'of', 'for', 'a', 'is', 'on', 'at', 'this', 'we', 'are', 'you', 'with', 'from', 'it'}
filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

# Count frequencies
word_counts = Counter(filtered_words)
common_words = word_counts.most_common(15) # Get top 15 words

words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])

plt.figure(figsize=(10, 6))
plt.bar(words_df['Word'], words_df['Frequency'], color='#8b0000', edgecolor='black')
plt.title('Top 15 Most Frequent Words in HIGH Urgency Tweets', fontsize=14, fontweight='bold')
plt.xlabel('Words', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
words_path = os.path.join(VISUALS_PATH, 'high_urgency_words.png')
plt.tight_layout()
plt.savefig(words_path, dpi=300)
print(f"✅ Saved chart to: {words_path}")

print("\n🚀 Visualizations Complete! Check the 'visuals' folder.")