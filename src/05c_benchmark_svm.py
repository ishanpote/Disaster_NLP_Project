import pickle
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = "../data/processed/"
MODELS_PATH = "../models/"

# --- 1. LOAD BALANCED FEATURES ---
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
    print("❌ Error: Feature files not found.")
    exit()

# --- 2. TRAIN SVM MODEL ---
print("\nTraining Benchmark Model (Linear Support Vector Machine)...")
# LinearSVC is incredibly fast and optimized for text classification
svm_model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
svm_model.fit(X_train, y_train)
print("✅ SVM Training Complete.")

# --- 3. EVALUATE SVM ---
print("\nEvaluating SVM on Real-World Test Set...")
y_pred_svm = svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"\n🏆 SVM Overall Accuracy: {accuracy_svm:.2%}")
print("\n--- Detailed Classification Report (SVM) ---")
print(classification_report(y_test, y_pred_svm))

# --- 4. SAVE MODEL (Optional) ---
with open(os.path.join(MODELS_PATH, "svm_model.pkl"), "wb") as f:
    pickle.dump(svm_model, f)
print("\n🚀 Benchmark complete. SVM model saved.")