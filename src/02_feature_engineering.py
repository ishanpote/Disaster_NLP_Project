import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = "E:\\3rd_year\\6SEM\\Project\\Disaster_NLP_Project\\data\\processed"
MODELS_PATH = "E:\\3rd_year\\6SEM\\Project\\Disaster_NLP_Project\\models"

# Ensure models directory exists
os.makedirs(MODELS_PATH, exist_ok=True)

# --- 1. LOAD DATA ---
print("Loading processed data...")
try:
    df_humaid = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv"))
    # Drop rows where text might be empty after cleaning
    df_humaid = df_humaid.dropna(subset=['cleaned_text'])
    print(f"✅ Loaded HumAID Data: {len(df_humaid)} rows")
except FileNotFoundError:
    print("❌ Error: humaid_cleaned.csv not found. Run 01_preprocessing.py first.")
    exit()

# --- 2. SPLIT DATA (TRAIN vs TEST) ---
# We need to hide some data from the model to test it later
print("\nSplitting data into Training and Testing sets...")
X = df_humaid['cleaned_text']
y = df_humaid['urgency']

# 80% for Training, 20% for Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train Set: {len(X_train)} tweets")
print(f"Test Set:  {len(X_test)} tweets")

# --- 3. TF-IDF VECTORIZATION ---
print("\nVectorizing text (Converting to numbers)...")

# Max_features=5000 means we only keep the top 5,000 most important words
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

# Learn vocabulary from Train data ONLY (to prevent data leakage)
X_train_tfidf = tfidf.fit_transform(X_train)
# Just transform the Test data
X_test_tfidf = tfidf.transform(X_test)

print(f"✅ Vectorization Complete. Shape: {X_train_tfidf.shape}")

# --- 4. SAVE EVERYTHING ---
print("\nSaving data and vectorizer...")

# Save the Vectorizer (We need this to process new tweets later!)
with open(os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

# Save the transformed data (So we don't have to redo this every time)
with open(os.path.join(PROCESSED_DATA_PATH, "X_train_tfidf.pkl"), "wb") as f:
    pickle.dump(X_train_tfidf, f)
with open(os.path.join(PROCESSED_DATA_PATH, "X_test_tfidf.pkl"), "wb") as f:
    pickle.dump(X_test_tfidf, f)
with open(os.path.join(PROCESSED_DATA_PATH, "y_train.pkl"), "wb") as f:
    pickle.dump(y_train, f)
with open(os.path.join(PROCESSED_DATA_PATH, "y_test.pkl"), "wb") as f:
    pickle.dump(y_test, f)

print("🚀 Step 2 Complete! Features saved in 'models/' and 'data/processed/'")