import pandas as pd
import os
import pickle
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download the tokenizer if not already present
nltk.download('punkt', quiet=True)

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = "../data/processed/"
MODELS_PATH = "../models/"

print("Loading cleaned text data for Word2Vec training...")
try:
    # We need the actual text, not the TF-IDF vectors, so we load the cleaned CSV
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv")).dropna(subset=['cleaned_text'])
    print("✅ Data Loaded.")
except FileNotFoundError:
    print("❌ Error: Cleaned data file not found.")
    exit()

# --- 1. PREPARE TEXT FOR WORD2VEC ---
print("\nTokenizing sentences (breaking tweets into individual words)...")
# Word2Vec requires a list of lists: [['the', 'bridge', 'collapsed'], ['send', 'help', 'now']]
sentences = [str(text).split() for text in df['cleaned_text']]

# --- 2. TRAIN THE WORD2VEC MODEL ---
print("\nTraining Word2Vec Embeddings (Teaching the AI semantic meaning)...")
# vector_size=100: Each word gets 100 coordinates to define its meaning
# window=5: Looks at 5 words before and after to understand context
# min_count=2: Ignores typos or words that only appear once
w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4)

print("✅ Word2Vec Training Complete!")

# Let's test the AI's new brain!
print("\n--- AI Semantic Understanding Test ---")
test_word = "water"
if test_word in w2v_model.wv:
    similar_words = w2v_model.wv.most_similar(test_word, topn=5)
    print(f"Words most mathematically similar to '{test_word}':")
    for word, score in similar_words:
        print(f" - {word} (Similarity: {score:.2f})")
else:
    print(f"Word '{test_word}' not found in vocabulary.")

# --- 3. SAVE THE MODEL ---
# We save the actual Word2Vec model so the Deep Learning network can use it later
model_path = os.path.join(MODELS_PATH, "word2vec_disaster.model")
w2v_model.save(model_path)
print(f"\n💾 Semantic dictionary saved to {model_path}!")