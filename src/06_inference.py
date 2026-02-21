import pickle
import re
import os

# --- CONFIGURATION ---
MODELS_PATH = "E:\\3rd_year\\6SEM\\Project\\Disaster_NLP_Project\\models"

# --- 1. LOAD MODEL & VECTORIZER ---
print("Loading Model and Vectorizer...")
try:
    with open(os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(MODELS_PATH, "logistic_regression_model.pkl"), "rb") as f:
        model = pickle.load(f)
    print("✅ System Ready for Live Inference!\n")
    print("-" * 50)
except FileNotFoundError:
    print("❌ Error: Model files not found in the 'models' folder.")
    exit()

# --- 2. TEXT CLEANING (Must match Phase 1) ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. LIVE DEMO LOOP ---
print("Type a simulated tweet below (or type 'exit' to quit):")
while True:
    user_input = input("\n📝 Enter Tweet: ")
    if user_input.lower() == 'exit':
        print("Ending demo. Goodbye!")
        break
    
    # Process the input
    cleaned_input = clean_text(user_input)
    vectorized_input = tfidf.transform([cleaned_input]) # MUST be an array
    
    # Predict Class and Probabilities
    prediction = model.predict(vectorized_input)[0]
    probabilities = model.predict_proba(vectorized_input)[0]
    classes = model.classes_
    
    # Display Results
    print(f"🚨 AI Classification: **{prediction.upper()} URGENCY**")
    print("Confidence Scores:")
    for cls, prob in zip(classes, probabilities):
        print(f"  - {cls}: {prob:.1%}")
    print("-" * 50)