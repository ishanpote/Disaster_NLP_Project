import pandas as pd
import numpy as np
import os
import pickle
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "../data/processed/")
MODELS_PATH = os.path.join(BASE_DIR, "../models/")

print("Loading cleaned dataset...")
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv")).dropna(subset=['cleaned_text'])

# To keep processing time reasonable on a local machine, we will sample the dataset
# If you have a powerful GPU, you can remove the .sample() to run on the full dataset
print("Sampling data for local BERT processing...")
df_sample = df.sample(n=10000, random_state=42) 
X_text = df_sample['cleaned_text'].astype(str).tolist()
y = df_sample['urgency'].tolist()

# --- 1. LOAD PRE-TRAINED BERT ---
print("\nDownloading/Loading Pre-trained DistilBERT Model from Hugging Face...")
# We use DistilBERT because it is 60% faster but retains 97% of BERT's language understanding
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Check for GPU (CUDA) to speed up processing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = bert_model.to(device)
print(f"✅ BERT Model loaded successfully on: {device}")

# --- 2. EXTRACT CONTEXTUAL EMBEDDINGS ---
print("\nTranslating tweets into 768-dimensional BERT vectors (This will take a few minutes)...")
def get_bert_embeddings(texts):
    # Tokenize the text and pad it to the exact same length
    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
    
    with torch.no_grad(): # Disable gradient calculation for feature extraction
        output = bert_model(**encoded_input)
    
    # We only take the [CLS] token (the first token), which acts as a summary of the entire sentence
    return output.last_hidden_state[:, 0, :].cpu().numpy()

# Process in batches to prevent computer crashes from memory overload
batch_size = 100
X_features = []
total_batches = len(X_text) // batch_size + 1

for i in range(0, len(X_text), batch_size):
    batch_texts = X_text[i:i+batch_size]
    batch_embeddings = get_bert_embeddings(batch_texts)
    X_features.append(batch_embeddings)
    if (i // batch_size) % 10 == 0:
        print(f"Processed batch {i // batch_size} of {total_batches}...")

X_features = np.vstack(X_features)
print("✅ Contextual Embeddings Generated!")

# --- 3. TRAIN PHASE 1 CHAMPION (LOGISTIC REGRESSION) ---
print("\nTraining Phase 1 Champion model using new BERT Brain...")
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)

# We use class_weight='balanced' instead of SMOTE for cleaner architecture
log_reg = LogisticRegression(max_iter=2000, class_weight='balanced')
log_reg.fit(X_train, y_train)

# --- 4. EVALUATE ---
print("\nEvaluating BERT + Logistic Regression Hybrid...")
y_pred = log_reg.predict(X_test)

print(f"\n🚀 BERT Hybrid Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

# Save the model and the embeddings for later
with open(os.path.join(MODELS_PATH, "bert_logreg_model.pkl"), "wb") as f:
    pickle.dump(log_reg, f)
print("\n💾 Model saved successfully!")