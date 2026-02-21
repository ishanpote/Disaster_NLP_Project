import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = "E:\\3rd_year\\6SEM\\Project\\Disaster_NLP_Project\\data\\processed"
MODELS_PATH = "../models/"

# --- 1. LOAD FEATURES (Notice we are loading the SMOTE training data!) ---
print("Loading balanced features and labels...")
try:
    with open(os.path.join(PROCESSED_DATA_PATH, "X_train_tfidf_smote.pkl"), "rb") as f:
        X_train = pickle.load(f)
    # We test on the ORIGINAL, untouched test set
    with open(os.path.join(PROCESSED_DATA_PATH, "X_test_tfidf.pkl"), "rb") as f:
        X_test = pickle.load(f)
        
    with open(os.path.join(PROCESSED_DATA_PATH, "y_train_smote.pkl"), "rb") as f:
        y_train = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)
    print("✅ Balanced Data Loaded Successfully.")
except FileNotFoundError:
    print("❌ Error: SMOTE feature files not found.")
    exit()

# --- 2. TRAIN MODEL (Logistic Regression) ---
print("\nTraining Baseline Model (Logistic Regression)...")
# We no longer need 'class_weight=balanced' because we physically balanced the data!
model = LogisticRegression(C=1.0, max_iter=1000, multi_class='ovr')
model.fit(X_train, y_train)
print("✅ Model Training Complete.")

# --- 3. EVALUATE MODEL ---
print("\nEvaluating on Real-World Test Set...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🏆 Overall Model Accuracy: {accuracy:.2%}")

print("\n--- Detailed Classification Report ---")
# This prints Precision, Recall, and F1-Score as required by your Problem Statement
print(classification_report(y_test, y_pred))

# --- 4. SAVE MODEL ---
print("\nSaving trained model...")
with open(os.path.join(MODELS_PATH, "logistic_regression_model.pkl"), "wb") as f:
    pickle.dump(model, f)

print(f"🚀 Step 3 Complete! Model saved to {MODELS_PATH}")