import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lime.lime_text import LimeTextExplainer

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "../data/processed/")
MODELS_PATH = os.path.join(BASE_DIR, "../models/")
ROBERTA_DIR = os.path.join(MODELS_PATH, "roberta_urgency_classifier")

print("🔍 Initializing XAI Diagnostic Tool (LIME)...")

# --- 1. LOAD DATA & FIND A MISTAKE ---
print("Loading test data to find a misclassified tweet...")
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv")).dropna(subset=['cleaned_text', 'urgency'])

X = df['cleaned_text'].astype(str).tolist()
y = df['urgency'].tolist()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
classes = label_encoder.classes_

_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)

# --- 2. LOAD ROBERTA ---
print("Loading Fine-Tuned RoBERTa Brain...")
tokenizer = AutoTokenizer.from_pretrained(ROBERTA_DIR)
model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_DIR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# --- 3. LIME PREDICTION FUNCTION ---
# LIME needs a specific function that takes raw text and returns probabilities
def predictor_wrapper(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Convert logits to probabilities
    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
    return probs

# --- 4. HUNTING FOR A SPECIFIC BLIND SPOT ---
print("Hunting for a tweet where RoBERTa confused 'High' and 'Medium' urgency...")
# We will predict the first 200 test tweets just to find a good mistake to analyze
test_subset = X_test
true_subset = y_test
subset_probs = predictor_wrapper(test_subset)
subset_preds = np.argmax(subset_probs, axis=1)

target_idx = -1
for i in range(len(test_subset)):
    true_label = classes[true_subset[i]]
    pred_label = classes[subset_preds[i]]
    
    # Let's find a case where it was actually High urgency, but the AI said Medium
    if true_label == 'High' and pred_label == 'Medium':
        target_idx = i
        break

if target_idx == -1:
    print("Could not find a High->Medium misclassification in the first 200 tweets. Just analyzing the first tweet instead.")
    target_idx = 0

text_to_explain = test_subset[target_idx]
actual_label = classes[true_subset[target_idx]]
predicted_label = classes[subset_preds[target_idx]]

print("\n" + "="*50)
print(f"🚨 FOUND TARGET FOR DIAGNOSTICS:")
print(f"Text: '{text_to_explain}'")
print(f"Human labeled it: {actual_label}")
print(f"RoBERTa guessed: {predicted_label}")
print("="*50)

# --- 5. GENERATE THE LIME EXPLANATION ---
print("\nGenerating visual explanation... (Running 500 perturbations)")
explainer = LimeTextExplainer(class_names=classes)

# We ask LIME to explain the prediction for the specific label RoBERTa *actually* guessed
exp = explainer.explain_instance(
    text_to_explain, 
    predictor_wrapper, 
    num_features=10, 
    num_samples=500,
    top_labels=1
)

# --- 6. EXPORT VISUALIZATION ---
output_path = os.path.join(BASE_DIR, "lime_explanation.html")
exp.save_to_file(output_path)
print(f"\n✅ Diagnostic Complete! Open this file in your web browser to see the AI's brain:")
print(f"➡️ {output_path}")