import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = "E:\\3rd_year\\6SEM\\Project\\Disaster_NLP_Project\\data\\processed"
MODELS_PATH = "E:\\3rd_year\\6SEM\\Project\\Disaster_NLP_Project\\models"

# --- 1. LOAD BALANCED DATA ---
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

# --- 2. DEFINE THE GRID SEARCH ---
print("\nSetting up Grid Search for Logistic Regression...")
# We are testing 3 different regularization strengths and 2 different solvers
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'max_iter': [1000, 2000]
}

# Initialize the base model
base_model = LogisticRegression(random_state=42)

# Set up the Grid Search (cv=3 means it will triple-check every single combination)
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# --- 3. TRAIN AND TUNE ---
print("Starting the tuning process (this might take a few minutes)...")
grid_search.fit(X_train, y_train)

print(f"\n✅ Tuning Complete!")
print(f"🏆 Best Mathematical Parameters Found: {grid_search.best_params_}")

# --- 4. EVALUATE THE OPTIMIZED MODEL ---
best_model = grid_search.best_estimator_
print("\nEvaluating Optimized Model on Test Set...")
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n🚀 Optimized Overall Accuracy: {acc:.2%}")
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

# Save the absolute best model to overwrite the old one
with open(os.path.join(MODELS_PATH, "logistic_regression_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
print("\n💾 Optimized model saved successfully!")