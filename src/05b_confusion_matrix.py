import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = "E:\3rd_year\6SEM\Project\Disaster_NLP_Project\data\processed"
MODELS_PATH = "E:\3rd_year\6SEM\Project\Disaster_NLP_Project\models"
VISUALS_PATH = "E:\3rd_year\6SEM\Project\Disaster_NLP_Project\visuals"
os.makedirs(VISUALS_PATH, exist_ok=True)

# --- 1. LOAD TEST DATA & MODEL ---
print("Loading test data and trained model...")
try:
    with open(os.path.join(PROCESSED_DATA_PATH, "X_test_tfidf.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)
    with open(os.path.join(MODELS_PATH, "logistic_regression_model.pkl"), "rb") as f:
        model = pickle.load(f)
    print("✅ Model and Data Loaded.")
except FileNotFoundError:
    print("❌ Error: Missing files. Ensure you have run scripts 02 and 05.")
    exit()

# --- 2. GENERATE PREDICTIONS ---
print("Making predictions on the test set...")
y_pred = model.predict(X_test)
classes = model.classes_

# --- 3. CREATE CONFUSION MATRIX ---
print("Generating visual Confusion Matrix...")
# This calculates the exact numbers of true positives, false positives, etc.
cm = confusion_matrix(y_test, y_pred, labels=classes)

# --- 4. PLOT AND SAVE ---
plt.figure(figsize=(8, 6))
# We use 'Blues' color map. annot=True puts the numbers inside the boxes, fmt='d' makes them integers
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes, 
            linewidths=1, linecolor='black')

plt.title("Confusion Matrix: Logistic Regression (Baseline)", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Urgency (AI Guess)", fontsize=12, fontweight='bold')
plt.ylabel("Actual Urgency (True Label)", fontsize=12, fontweight='bold')
plt.tight_layout()

save_path = os.path.join(VISUALS_PATH, "baseline_confusion_matrix.png")
plt.savefig(save_path, dpi=300)
print(f"\n🚀 Success! Confusion Matrix saved to: {save_path}")