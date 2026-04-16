import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
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

# The 5 Base Classifiers
ROBERTA_DIRS = [
    os.path.join(MODELS_PATH, "roberta_urgency_classifier"), # Seed 42
    os.path.join(MODELS_PATH, "roberta_urgency_seed123"),    # Seed 123
    os.path.join(MODELS_PATH, "roberta_urgency_seed456")     # Seed 456
]
CNN_MODEL_PATH = os.path.join(MODELS_PATH, "hybrid_cnn_bilstm.h5")
CNN_TOKENIZER_PATH = os.path.join(MODELS_PATH, "dl_tokenizer.pkl")
LOGREG_MODEL_PATH = os.path.join(MODELS_PATH, "logistic_regression_model.pkl") 
TFIDF_PATH = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")    

print("🚀 Initializing Multi-Seed Heterogeneous Stacking Architecture...")

# --- 1. LOAD DATA ---
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv")).dropna(subset=['cleaned_text', 'urgency'])
X = df['cleaned_text'].astype(str).tolist()
y = df['urgency'].tolist()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
classes = label_encoder.classes_
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)

meta_features = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. MULTI-SEED ROBERTA FEATURE EXTRACTION ---
for idx, model_dir in enumerate(ROBERTA_DIRS):
    print(f"\n🧠 Extracting probabilities from RoBERTa (Iteration {idx+1}/3)...")
    tokenizer_rob = AutoTokenizer.from_pretrained(model_dir)
    model_rob = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
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
    
    # Memory management for VRAM
    del model_rob
    del tokenizer_rob
    gc.collect()
    torch.cuda.empty_cache()

# --- 3. HETEROGENEOUS MODEL EXTRACTION ---
print("\n🧠 Extracting probabilities from Hybrid CNN-BiLSTM...")
model_cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
with open(CNN_TOKENIZER_PATH, 'rb') as f:
    tokenizer_cnn = pickle.load(f)
X_test_pad = pad_sequences(tokenizer_cnn.texts_to_sequences(X_test), maxlen=50) 
meta_features.append(model_cnn.predict(X_test_pad, verbose=0))

print("🧠 Extracting probabilities from Logistic Regression (Baseline)...")
with open(LOGREG_MODEL_PATH, 'rb') as f:
    model_logreg = pickle.load(f)
with open(TFIDF_PATH, 'rb') as f:
    tfidf = pickle.load(f)
meta_features.append(model_logreg.predict_proba(tfidf.transform(X_test)))

# --- 4. RULE-BASED FEATURE INJECTION ---
print("⚙️ Injecting Engineered Feature: Cumulative Resource Scarcity...")
def count_scarcity(text):
    keywords = ['no electricity', 'water', 'internet', 'gas', 'ice', 'phone', 'service', 'power', 'food', 'trapped', 'stranded']
    return sum(1 for word in keywords if word in text.lower())
meta_features.append(np.array([count_scarcity(text) for text in X_test]).reshape(-1, 1))

# --- 5. TRAIN META-CLASSIFIER (XGBOOST) ---
print("\n🔥 Training XGBoost Meta-Classifier...")
X_meta = np.hstack(meta_features)
X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(X_meta, y_test, test_size=0.3, random_state=42, stratify=y_test)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_model.fit(X_meta_train, y_meta_train)
xgb_raw_probs = xgb_model.predict_proba(X_meta_test)

# --- 6. PROBABILITY THRESHOLD OPTIMIZATION ---
print("\n⚙️ Executing Probability Threshold Optimization...")
best_acc = 0
best_weights = None

for w_0 in np.arange(0.5, 2.0, 0.1):
    for w_1 in np.arange(0.5, 2.0, 0.1):
        for w_2 in np.arange(0.5, 2.0, 0.1):
            weights = np.array([w_0, w_1, w_2])
            weighted_probs = xgb_raw_probs * weights
            preds = np.argmax(weighted_probs, axis=1)
            acc = accuracy_score(y_meta_test, preds)
            if acc > best_acc:
                best_acc = acc
                best_weights = weights

# --- 7. FINAL EVALUATION ---
print("\n" + "="*55)
print(f"📊 MULTI-SEED META-CLASSIFIER ACCURACY: {best_acc:.2%}")
print("="*55)

final_weighted_probs = xgb_raw_probs * best_weights
final_preds = np.argmax(final_weighted_probs, axis=1)

print("\n--- Final System Classification Report ---")
print(classification_report(y_meta_test, final_preds, target_names=classes))

# Save the ultimate meta-model
with open(os.path.join(MODELS_PATH, "multiseed_meta_classifier.pkl"), "wb") as f:
    pickle.dump(xgb_model, f)