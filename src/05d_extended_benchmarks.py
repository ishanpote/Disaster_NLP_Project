import pickle
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = "E:\\3rd_year\\6SEM\\Project\\Disaster_NLP_Project\\data\\processed"
MODELS_PATH = "E:\\3rd_year\\6SEM\\Project\\Disaster_NLP_Project\\models"

# --- 1. LOAD BALANCED BIGRAM FEATURES ---
print("Loading balanced features and labels...")
try:
    with open(os.path.join(PROCESSED_DATA_PATH, "X_train_tfidf_smote.pkl"), "rb") as f:
        X_train = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "X_test_tfidf.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "y_train_smote.pkl"), "rb") as f:
        y_train = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)
    print("✅ Data Loaded.")
except FileNotFoundError:
    print("❌ Error: Feature files not found. Please run the feature engineering pipeline.")
    exit()

# --- 2. TRAIN MULTINOMIAL NAIVE BAYES ---
print("\n" + "="*50)
print("Training Model 1: Multinomial Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

print("Evaluating Naive Bayes...")
y_pred_nb = nb_model.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)
print(f"🏆 Naive Bayes Overall Accuracy: {acc_nb:.2%}")
print("Classification Report (Naive Bayes):\n")
print(classification_report(y_test, y_pred_nb))

# --- 3. TRAIN RANDOM FOREST ---
print("\n" + "="*50)
print("Training Model 2: Random Forest (This might take a minute)...")
# We use 100 trees (n_estimators=100) and n_jobs=-1 to use all CPU cores for speed
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("Evaluating Random Forest...")
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"🏆 Random Forest Overall Accuracy: {acc_rf:.2%}")
print("Classification Report (Random Forest):\n")
print(classification_report(y_test, y_pred_rf))

print("\n" + "="*50)
print("🚀 Extended Benchmarks Complete!")