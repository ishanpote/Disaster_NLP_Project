import pandas as pd
import os
import pickle
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download the tokenizer if not already present
nltk.download('punkt', quiet=True)

# --- CONFIGURATION (BULLETPROOF PATHS) ---
# This anchors the paths to the exact location of this python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "../data/processed/")
MODELS_PATH = os.path.join(BASE_DIR, "../models/")

print("Loading cleaned text data for Word2Vec training...")
try:
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv")).dropna(subset=['cleaned_text'])
    print("✅ Data Loaded.")
except FileNotFoundError:
    print(f"❌ Error: Could not find humaid_cleaned.csv in {os.path.abspath(PROCESSED_DATA_PATH)}")
    exit()

# --- 1. PREPARE TEXT FOR WORD2VEC ---
print("\nTokenizing sentences (breaking tweets into individual words)...")
# Word2Vec requires a list of lists: [['the', 'bridge', 'collapsed'], ['send', 'help', 'now']]
sentences = [str(text).split() for text in df['cleaned_text']]

# --- 2. TRAIN THE WORD2VEC MODEL ---
print("\nTraining Word2Vec Embeddings (Teaching the AI semantic meaning)...")
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
model_path = os.path.join(MODELS_PATH, "word2vec_disaster.model")
w2v_model.save(model_path)
print(f"\n💾 Semantic dictionary saved to {os.path.abspath(model_path)}!")