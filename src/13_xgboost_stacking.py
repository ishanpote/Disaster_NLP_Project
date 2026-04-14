import os
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

ROBERTA_DIR = os.path.join(MODELS_PATH, "roberta_urgency_classifier")
CNN_MODEL_PATH = os.path.join(MODELS_PATH, "hybrid_cnn_bilstm.h5")
CNN_TOKENIZER_PATH = os.path.join(MODELS_PATH, "dl_tokenizer.pkl")
LOGREG_MODEL_PATH = os.path.join(MODELS_PATH, "logistic_regression_model.pkl") 
TFIDF_PATH = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")    

print("🚀 Initializing Kaggle-Style Stacking Architecture...")

# --- 1. LOAD DATA ---
print("Loading the test dataset...")
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv")).dropna(subset=['cleaned_text', 'urgency'])

X = df['cleaned_text'].astype(str).tolist()
y = df['urgency'].tolist()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)

# --- 2. EXTRACT PREDICTIONS (THE BASE MODELS) ---
# We need to collect the probability outputs from all 3 models to feed into XGBoost
meta_features = []

# RoBERTa Probabilities
print("\n🧠 Extracting probabilities from RoBERTa...")
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

# CNN-BiLSTM Probabilities
print("🧠 Extracting probabilities from CNN-BiLSTM...")
model_cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
with open(CNN_TOKENIZER_PATH, 'rb') as f:
    tokenizer_cnn = pickle.load(f)
X_test_seq = tokenizer_cnn.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=50) 
cnn_probs = model_cnn.predict(X_test_pad, verbose=0)
meta_features.append(cnn_probs)

# Logistic Regression Probabilities
print("🧠 Extracting probabilities from Logistic Regression...")
with open(LOGREG_MODEL_PATH, 'rb') as f:
    model_logreg = pickle.load(f)
with open(TFIDF_PATH, 'rb') as f:
    tfidf = pickle.load(f)
X_test_tfidf = tfidf.transform(X_test)
logreg_probs = model_logreg.predict_proba(X_test_tfidf)
meta_features.append(logreg_probs)

# --- 3. THE LIME FIX: RESOURCE SCARCITY ENGINEERED FEATURE ---
print("⚙️ Injecting Rule-Based Feature: Resource Scarcity Count...")
def count_scarcity(text):
    # Keywords that indicate cascading infrastructure failure
    keywords = ['no electricity', 'water', 'internet', 'gas', 'ice', 'phone', 'service', 'power', 'food', 'trapped', 'stranded']
    text_lower = text.lower()
    return sum(1 for word in keywords if word in text_lower)

scarcity_counts = np.array([count_scarcity(text) for text in X_test]).reshape(-1, 1)
meta_features.append(scarcity_counts)

# Combine everything into the final Meta-Dataset (10 columns: 3+3+3+1)
X_meta = np.hstack(meta_features)

# --- 4. TRAIN THE META-MODEL (XGBOOST) ---
print("\n🔥 Training XGBoost Meta-Model...")
# We split the test set into a meta-train and meta-test to evaluate XGBoost fairly
X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(X_meta, y_test, test_size=0.3, random_state=42, stratify=y_test)

# Configure XGBoost to correct the mistakes of the base models
xgb_model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)

xgb_model.fit(X_meta_train, y_meta_train)

# --- 5. FINAL EVALUATION ---
print("\nEvaluating the Ultimate Stacking Architecture...")
y_pred_meta = xgb_model.predict(X_meta_test)

print("\n" + "="*50)
print(f"🏆 XGBOOST STACKING ACCURACY: {accuracy_score(y_meta_test, y_pred_meta):.2%}")
print("="*50)

print("\n--- Detailed Classification Report ---")
classes = label_encoder.classes_
print(classification_report(y_meta_test, y_pred_meta, target_names=classes))

# Save the ultimate meta-model
with open(os.path.join(MODELS_PATH, "xgboost_meta_model.pkl"), "wb") as f:
    pickle.dump(xgb_model, f)
print("\n💾 Meta-Model saved successfully!")