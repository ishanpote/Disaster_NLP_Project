import pickle
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = "E:\\3rd_year\\6SEM\\Project\\Disaster_NLP_Project\\data\\processed"

# --- 1. LOAD THE IMBALANCED FEATURES ---
print("Loading TF-IDF features and labels...")
try:
    with open(os.path.join(PROCESSED_DATA_PATH, "X_train_tfidf.pkl"), "rb") as f:
        X_train = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)
    print("✅ Data Loaded Successfully.")
except FileNotFoundError:
    print("❌ Error: Feature files not found. Run 02_feature_engineering.py first.")
    exit()

# --- 2. SHOW BEFORE STATE ---
print("\n--- Class Distribution BEFORE SMOTE ---")
counter_before = Counter(y_train)
for urgency, count in counter_before.items():
    print(f"{urgency}: {count}")

# --- 3. APPLY SMOTE ---
print("\nApplying SMOTE (Synthesizing new data points)...")
# random_state ensures we get the same results every time we run it
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# --- 4. SHOW AFTER STATE ---
print("\n--- Class Distribution AFTER SMOTE ---")
counter_after = Counter(y_train_smote)
for urgency, count in counter_after.items():
    print(f"{urgency}: {count}")

# --- 5. SAVE THE BALANCED DATA ---
print("\nSaving balanced dataset...")
with open(os.path.join(PROCESSED_DATA_PATH, "X_train_tfidf_smote.pkl"), "wb") as f:
    pickle.dump(X_train_smote, f)
with open(os.path.join(PROCESSED_DATA_PATH, "y_train_smote.pkl"), "wb") as f:
    pickle.dump(y_train_smote, f)

print("🚀 Step 2b Complete! Data is perfectly balanced and ready for the Baseline Model.")