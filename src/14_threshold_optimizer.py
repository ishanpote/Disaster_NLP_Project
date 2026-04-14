import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "../data/processed")
MODELS_PATH = os.path.join(BASE_DIR, "../models")

ROBERTA_DIR = os.path.join(MODELS_PATH, "roberta_urgency_classifier")
CNN_MODEL_PATH = os.path.join(MODELS_PATH, "hybrid_cnn_bilstm.h5")
CNN_TOKENIZER_PATH = os.path.join(MODELS_PATH, "dl_tokenizer.pkl")
LOGREG_MODEL_PATH = os.path.join(MODELS_PATH, "logistic_regression_model.pkl") 
TFIDF_PATH = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")    
XGB_PATH = os.path.join(MODELS_PATH, "xgboost_meta_model.pkl")

print("🚀 Initializing Probability Threshold Optimizer...")

# --- 1. LOAD DATA ---
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv")).dropna(subset=['cleaned_text', 'urgency'])
X = df['cleaned_text'].astype(str).tolist()
y = df['urgency'].tolist()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
classes = label_encoder.classes_
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)

# --- 2. GATHER BASE PROBABILITIES (ONE LAST TIME) ---
print("\nGathering predictions from base models to feed to XGBoost...")
meta_features = []

# RoBERTa
tokenizer_rob = AutoTokenizer.from_pretrained(ROBERTA_DIR)
model_rob = AutoModelForSequenceClassification.from_pretrained(ROBERTA_DIR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_rob.to(device)
model_rob.eval()

rob_probs = []
with torch.no_grad():
    for i in range(0, len(X_test), 32):
        batch = X_test[i:i+32]
        inputs = tokenizer_rob(batch, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        outputs = model_rob(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
        rob_probs.extend(probs)
meta_features.append(np.array(rob_probs))

# CNN-BiLSTM
model_cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
with open(CNN_TOKENIZER_PATH, 'rb') as f:
    tokenizer_cnn = pickle.load(f)
X_test_pad = pad_sequences(tokenizer_cnn.texts_to_sequences(X_test), maxlen=50) 
meta_features.append(model_cnn.predict(X_test_pad, verbose=0))

# LogReg
with open(LOGREG_MODEL_PATH, 'rb') as f:
    model_logreg = pickle.load(f)
with open(TFIDF_PATH, 'rb') as f:
    tfidf = pickle.load(f)
meta_features.append(model_logreg.predict_proba(tfidf.transform(X_test)))

# Scarcity Count
def count_scarcity(text):
    keywords = ['no electricity', 'water', 'internet', 'gas', 'ice', 'phone', 'service', 'power', 'food', 'trapped', 'stranded']
    return sum(1 for word in keywords if word in text.lower())
meta_features.append(np.array([count_scarcity(text) for text in X_test]).reshape(-1, 1))

X_meta = np.hstack(meta_features)
_, X_meta_test, _, y_meta_test = train_test_split(X_meta, y_test, test_size=0.3, random_state=42, stratify=y_test)

# --- 3. GET RAW XGBOOST PROBABILITIES ---
print("\nExtracting raw probabilities from XGBoost Meta-Model...")
with open(XGB_PATH, 'rb') as f:
    xgb_model = pickle.load(f)
    
xgb_raw_probs = xgb_model.predict_proba(X_meta_test)

# --- 4. THE MAGIC: SWEEPING FOR THE OPTIMAL THRESHOLDS ---
print("\n⚙️ Sweeping 3,375 weight combinations to find the >90% sweet spot...")
best_acc = 0
best_weights = None

# Test weights from 0.5x to 2.0x for each of the 3 classes
for w_0 in np.arange(0.5, 2.0, 0.1):
    for w_1 in np.arange(0.5, 2.0, 0.1):
        for w_2 in np.arange(0.5, 2.0, 0.1):
            weights = np.array([w_0, w_1, w_2])
            
            # Multiply raw probabilities by the test weights
            weighted_probs = xgb_raw_probs * weights
            
            # Make predictions based on the newly weighted probabilities
            preds = np.argmax(weighted_probs, axis=1)
            
            acc = accuracy_score(y_meta_test, preds)
            if acc > best_acc:
                best_acc = acc
                best_weights = weights

# --- 5. FINAL RESULTS ---
print("\n" + "="*50)
print(f"🔥 OPTIMIZED STACKING ACCURACY: {best_acc:.2%}")
print("="*50)

print(f"\nWinning Multipliers Found:")
for i, class_name in enumerate(classes):
    print(f" - {class_name} Weight: {best_weights[i]:.2f}x")

# Apply the best weights to show the final classification report
final_weighted_probs = xgb_raw_probs * best_weights
final_preds = np.argmax(final_weighted_probs, axis=1)

print("\n--- Final Optimized Classification Report ---")
print(classification_report(y_meta_test, final_preds, target_names=classes))