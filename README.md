
Disaster NLP Project is an end-to-end urgency classification system for disaster-related tweets. It maps tweets into three operational priorities:

- High: immediate life/safety risk
- Medium: important situational or infrastructure information
- Low: non-urgent context (sympathy, general updates, etc.)

The repository contains:

- Classical NLP baseline (TF-IDF + Logistic Regression) with benchmarks
- Deep learning models (Word2Vec + CNN-BiLSTM, RoBERTa)
- Ensemble and stacking experiments (weighted voting + XGBoost meta-model)
- Explainability and diagnostics (LIME, error analysis, confusion matrices, visual analytics)

## Project Layout

```
Disaster_NLP_Project/
|- data/
|  |- raw/
|  |  |- train.csv
|  |  |- events_set1/...
|  |- processed/
|     |- humaid_cleaned.csv
|     |- kaggle_cleaned.csv
|     |- X_train_tfidf.pkl
|     |- X_test_tfidf.pkl
|     |- X_train_tfidf_smote.pkl
|     |- y_train_smote.pkl
|     |- misclassified_tweets.csv
|     |- critical_misses.csv
|- models/
|  |- logistic_regression_model.pkl
|  |- tfidf_vectorizer.pkl
|  |- word2vec_disaster.model
|  |- hybrid_cnn_bilstm.h5
|  |- roberta_urgency_*/
|  |- xgboost_meta_model.pkl
|- src/
|  |- 01_preprocessing.py
|  |- 02_feature_engineering.py
|  |- ...
|- visuals/
|- README.md
```

## Environment Setup

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

`requirements.csv` is currently empty, so install the required packages directly:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn \
            textblob wordcloud gensim nltk tensorflow torch transformers \
            datasets xgboost lime
```

If `textblob` or `nltk` resources are missing at runtime, download corpora/tokenizers when prompted.

## Data Pipeline

### 1) Preprocessing

Run:

```bash
python src/01_preprocessing.py
```

What it does:

- Cleans tweet text (URLs, mentions, hashtags, punctuation, whitespace normalization)
- Processes Kaggle `train.csv` into `data/processed/kaggle_cleaned.csv`
- Processes HumAID TSV files into `data/processed/humaid_cleaned.csv`
- Maps HumAID class labels to urgency levels (`High`, `Medium`, `Low`)

### 2) Feature Engineering

Run:

```bash
python src/02_feature_engineering.py
python src/02b_handle_imbalance.py
```

What it does:

- Splits train/test data with stratification
- Builds TF-IDF features (`max_features=5000`, `ngram_range=(1,2)`)
- Applies SMOTE on the training set
- Saves vectorizer and feature artifacts for reuse

## Model Training and Evaluation

### Phase 1: Classical Baseline and Benchmarks

```bash
python src/05_train_model.py
python src/05b_confusion_matrix.py
python src/05c_benchmark_svm.py
python src/05d_extended_benchmarks.py
python src/05e_hyperparameter_tuning.py
python src/06_inference.py
python src/08_error_analysis.py
```

Main outputs:

- `models/logistic_regression_model.pkl`
- `data/processed/misclassified_tweets.csv`
- `data/processed/critical_misses.csv`
- baseline confusion matrix in `visuals/`

### Phase 2: Deep Learning

```bash
python src/07_word_embeddings.py
python src/09_train_hybrid.py
python src/10_bert_feature_extractor.py
```

Main outputs:

- `models/word2vec_disaster.model`
- `models/hybrid_cnn_bilstm.h5`
- `models/roberta_urgency_seed456/` (or configured output)

### Phase 3: Ensembles, Stacking, Optimization, XAI

```bash
python src/11_ensemble_voting.py
python src/13_xgboost_stacking.py
python src/14_threshold_optimizer.py
python src/15_multiseed_meta_classifier.py
python src/12_explainable_ai.py
```

Main outputs:

- `models/xgboost_meta_model.pkl`
- `models/multiseed_meta_classifier.pkl`
- `src/lime_explanation.html`

## Visualization Scripts

Run these for EDA and presentation visuals:

```bash
python src/03_visualizations.py
python src/04_advanced_visuals.py
python src/04b_deeper_insights.py
python visuals/generate_confusion_matrix.py
python visuals/generate_dataset_graph.py
```

## Known Notes and Limitations

- Several Phase 1 scripts use hardcoded absolute Windows paths (`E:\3rd_year\...`). If your local path differs, update constants at the top of those scripts.
- RoBERTa/CNN/LIME workflows are compute-intensive. GPU is recommended for training.
- LIME can still hit CUDA memory limits in constrained environments; batching and CPU fallback are practical mitigations.
- Large model files should be handled carefully in version control. `.gitattributes` already tracks selected `model.safetensors` files with Git LFS.

## Current Status

- Baseline classical pipeline: implemented and reproducible
- Deep learning + transformer pipeline: implemented
- Ensemble and stacking experiments: implemented
- Explainability and threshold optimization: implemented
- IoT integration: planned extension (not implemented yet)

## Suggested IoT Extension (Next Phase)

For future implementation, connect field sensors and geospatial streams to create a multimodal urgency engine:

- Ingest IoT signals (rainfall, river level, wind, seismic alerts) via MQTT/HTTP
- Fuse sensor anomalies with tweet urgency probabilities
- Trigger geofenced escalation when both textual and sensor confidence are high
- Store event traces for post-incident auditing