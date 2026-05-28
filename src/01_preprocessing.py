import pandas as pd
import glob
import re
import os

RAW_DATA_PATH = "../data/raw/"
PROCESSED_DATA_PATH = "../data/processed/"
HUMAID_FOLDER = "events_set1"  

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# --- 1. TEXT CLEANING FUNCTION ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()  # Lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions (@user)
    text = re.sub(r'#', '', text)  # Remove hashtag symbol (keep the word)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# --- 2. URGENCY MAPPING LOGIC ---
def map_urgency(label):
    # HIGH URGENCY: Immediate threat to life or specific call for help
    high_labels = [
        'injured_or_dead_people', 
        'missing_or_found_people', 
        'requests_or_urgent_needs', 
        'displaced_people_and_evacuations'
    ]
    
    # MEDIUM URGENCY: Important situational info, but not necessarily life-or-death right now
    medium_labels = [
        'infrastructure_and_utility_damage', 
        'caution_and_advice'
    ]
    
    if label in high_labels:
        return 'High'
    elif label in medium_labels:
        return 'Medium'
    else:
        return 'Low'  # Includes volunteering, sympathy, not_humanitarian

# --- 3. PROCESS KAGGLE DATA (Noise Filter) ---
print("Processing Kaggle Data...")
try:
    df_kaggle = pd.read_csv(os.path.join(RAW_DATA_PATH, "train.csv"))
    df_kaggle['cleaned_text'] = df_kaggle['text'].apply(clean_text)
    df_kaggle[['id', 'cleaned_text', 'target']].to_csv(
        os.path.join(PROCESSED_DATA_PATH, "kaggle_cleaned.csv"), index=False
    )
    print(f"✅ Saved kaggle_cleaned.csv ({len(df_kaggle)} rows)")
except FileNotFoundError:
    print("❌ Error: train.csv not found. Check your 'raw' folder.")

# --- 4. PROCESS HUMAID DATA (Classifier) ---
print("\nProcessing HumAID Data...")
search_path = os.path.join(RAW_DATA_PATH, HUMAID_FOLDER, "**/*.tsv")
all_files = glob.glob(search_path, recursive=True)

if all_files:
    df_humaid = pd.concat((pd.read_csv(f, sep='\t') for f in all_files), ignore_index=True)
    
    df_humaid['cleaned_text'] = df_humaid['tweet_text'].apply(clean_text)
    
    # Apply Urgency Mapping
    df_humaid['urgency'] = df_humaid['class_label'].apply(map_urgency)
    
    # Save final version
    df_humaid[['tweet_id', 'cleaned_text', 'class_label', 'urgency']].to_csv(
        os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv"), index=False
    )
    print(f"✅ Saved humaid_cleaned.csv ({len(df_humaid)} rows)")
    
    # Show distribution
    print("\n--- Urgency Distribution ---")
    print(df_humaid['urgency'].value_counts())
else:
    print(f"❌ Error: No TSV files found in {search_path}")

print("\n🚀 Step 1 Complete! Data is ready for training.")