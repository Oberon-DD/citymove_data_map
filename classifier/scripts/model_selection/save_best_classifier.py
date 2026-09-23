#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save Best Classifier Configuration for CITY-MOVE

After running compare_classifiers.py and tune_hyperparameters_mpnet.py,
use this script to save the best classifier and embedding model for use
in your main classification script.

This creates a reusable classifier that can be loaded in Spyder.

Usage:
    # In Spyder or any Python IDE:
    %run save_best_classifier.py
"""

import os
import joblib
import pandas as pd
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

# Folder with the training and holdout workbooks (classifier/data in the published
# repository); set CITYMOVE_CLASSIFIER_DATA to use another folder.
DATA_DIR = os.environ.get(
    "CITYMOVE_CLASSIFIER_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "data"),
)

# Input paths
TRAIN_DOCS_PATH = os.path.join(DATA_DIR, "train_docs.xlsx")
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "train_labels.xlsx")
TUNED_MODEL_PATH = os.path.join(DATA_DIR, "tuned_classifier_svm.pkl")
TUNING_RESULTS_PATH = os.path.join(DATA_DIR, "hyperparameter_tuning_results_svm.xlsx")

# Output paths
BEST_CLASSIFIER_PATH = os.path.join(DATA_DIR, "best_classifier.pkl")
BEST_EMBEDDING_MODEL_PATH = os.path.join(DATA_DIR, "best_embedding_model")
CONFIG_PATH = os.path.join(DATA_DIR, "classifier_config.txt")

# Best configuration (update these after tuning)
BEST_EMBEDDING_MODEL = 'all-mpnet-base-v2'

# =========================
# LABEL NORMALIZATION
# =========================

def normalize_label(label: str) -> str:
    """Canonicalize labels."""
    if not isinstance(label, str):
        return label
    
    l = label.strip()
    
    CANONICAL_MAP = {
        "air quality": "Air Quality",
        "airquality": "Air Quality",
        "ncd's": "NCD's",
        "ncds": "NCD's",
        "ncd": "NCD's",
        "recr. & sports environ.": "Recreational and Sports Environment",
        "recr & sports environ.": "Recreational and Sports Environment",
        "recreational and sports environment": "Recreational and Sports Environment",
        "transport pa": "Transport PA",
        "recreational pa": "Recreational PA",
        "total pa": "Total PA",
        "other": "Other",
        "active transport environment": "Active Transport Environment",
        "demographics & ses": "Demographics & SES",
        "demographics and ses": "Demographics & SES",
        "demographis & ses": "Demographics & SES",
        "demographics ses": "Demographics & SES",
    }
    
    return CANONICAL_MAP.get(l.lower(), l)


# =========================
# MAIN
# =========================

def main():
    print("\n" + "="*80)
    print("SAVING BEST CLASSIFIER CONFIGURATION")
    print("="*80)
    
    # Check if tuned model exists
    if os.path.exists(TUNED_MODEL_PATH):
        print(f"\n✓ Found tuned classifier: {TUNED_MODEL_PATH}")
        
        # Load tuned model
        best_clf = joblib.load(TUNED_MODEL_PATH)
        
        # Get parameters
        params = best_clf.get_params()
        print("\nTuned parameters:")
        for key in ['kernel', 'C', 'gamma', 'class_weight']:
            if key in params:
                print(f"  {key:20s}: {params[key]}")
        
        # Load tuning results to get score
        if os.path.exists(TUNING_RESULTS_PATH):
            summary = pd.read_excel(TUNING_RESULTS_PATH, sheet_name='Summary')
            best_score = summary['Best F1 Score'].iloc[0]
            print(f"\n✓ Best F1 Score: {best_score:.4f}")
        
    else:
        print(f"\n⚠ Tuned model not found: {TUNED_MODEL_PATH}")
        print("Using best parameters from compare_classifiers.py results:")
        print("  SVM (RBF, balanced)")
        
        # Create classifier with best known parameters
        best_clf = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        
        # Train on current data
        print("\nTraining classifier on current data...")
        train_docs = pd.read_excel(TRAIN_DOCS_PATH).iloc[:, 0].astype(str)
        train_labels = pd.read_excel(TRAIN_LABELS_PATH).iloc[:, 0].astype(str).map(normalize_label)
        
        # Get embeddings
        print(f"Loading embedding model: {BEST_EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(BEST_EMBEDDING_MODEL)
        X_train = embedding_model.encode(train_docs.tolist(), batch_size=32, show_progress_bar=True)
        
        # Label encoding
        labels_unique = sorted(train_labels.unique())
        label_to_idx = {lab: i for i, lab in enumerate(labels_unique)}
        y_train = train_labels.map(label_to_idx).values
        
        # Train
        best_clf.fit(X_train, y_train)
        print("✓ Classifier trained")
    
    # Save classifier
    joblib.dump(best_clf, BEST_CLASSIFIER_PATH)
    print(f"\n✓ Saved classifier to: {BEST_CLASSIFIER_PATH}")
    
    # Download and save embedding model
    print(f"\nDownloading/loading embedding model: {BEST_EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(BEST_EMBEDDING_MODEL)
    
    # Save embedding model
    embedding_model.save(BEST_EMBEDDING_MODEL_PATH)
    print(f"✓ Saved embedding model to: {BEST_EMBEDDING_MODEL_PATH}")
    
    # Save configuration text file
    config_text = f"""
CITY-MOVE Best Classifier Configuration
=========================================

Generated: {pd.Timestamp.now()}

CLASSIFIER:
-----------
Type: SVC (Support Vector Classifier)
Parameters:
"""
    
    params = best_clf.get_params()
    for key, value in sorted(params.items()):
        config_text += f"  {key}: {value}\n"
    
    config_text += f"""
EMBEDDING MODEL:
----------------
Model: {BEST_EMBEDDING_MODEL}
Dimensions: 768
Location: {BEST_EMBEDDING_MODEL_PATH}

FILES CREATED:
--------------
Classifier: {BEST_CLASSIFIER_PATH}
Embedding Model: {BEST_EMBEDDING_MODEL_PATH}
Configuration: {CONFIG_PATH}

USAGE IN YOUR MAIN SCRIPT:
--------------------------
import joblib
from sentence_transformers import SentenceTransformer

# Load classifier
clf = joblib.load(r"{BEST_CLASSIFIER_PATH}")

# Load embedding model
embedding_model = SentenceTransformer(r"{BEST_EMBEDDING_MODEL_PATH}")

# Extract embeddings
def extract_embeddings(docs):
    return embedding_model.encode(docs, batch_size=32, show_progress_bar=False)

# Predict
X = extract_embeddings(["Your text here"])
predictions = clf.predict(X)
probabilities = clf.predict_proba(X)
"""
    
    with open(CONFIG_PATH, 'w') as f:
        f.write(config_text)
    
    print(f"✓ Saved configuration to: {CONFIG_PATH}")
    
    print("\n" + "="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    
    print("\n✓ Files created:")
    print(f"  1. Classifier:       {BEST_CLASSIFIER_PATH}")
    print(f"  2. Embedding Model:  {BEST_EMBEDDING_MODEL_PATH}")
    print(f"  3. Configuration:    {CONFIG_PATH}")
    
    print("\n📝 Next steps:")
    print("  1. Copy the integration code below")
    print("  2. Paste into your main classification script")
    print("  3. Update the fit_everything() and predict_with_probs() functions")
    
    print("\n" + "="*80)
    print("INTEGRATION CODE FOR YOUR MAIN SCRIPT")
    print("="*80)
    print("""
# Add these at the top of your script
import joblib
from sentence_transformers import SentenceTransformer

# Paths to saved models
BEST_CLASSIFIER_PATH = r"{classifier_path}"
BEST_EMBEDDING_MODEL_PATH = r"{embedding_path}"

# Load models (do this once at startup)
print("Loading best classifier and embedding model...")
BEST_CLF = joblib.load(BEST_CLASSIFIER_PATH)
EMBEDDING_MODEL = SentenceTransformer(BEST_EMBEDDING_MODEL_PATH)
print("✓ Models loaded")

# Function to extract embeddings
def extract_embeddings_mpnet(docs):
    \"\"\"Extract embeddings using all-mpnet-base-v2.\"\"\"
    return EMBEDDING_MODEL.encode(
        docs, 
        batch_size=32, 
        show_progress_bar=False
    )

# Update build_models() function:
def build_models():
    \"\"\"Build BERTopic (for interpretability) and use pre-trained SVM.\"\"\"
    
    # BERTopic for interpretability only
    empty_dimensionality_model = BaseDimensionalityReduction()
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    topic_model = BERTopic(
        umap_model=empty_dimensionality_model,
        ctfidf_model=ctfidf_model
    )
    
    # Use pre-trained classifier
    clf = BEST_CLF
    
    return topic_model, clf

# Update fit_everything() function:
def fit_everything(topic_model, clf, train_docs, train_labels, label_to_idx):
    \"\"\"Fit both models.\"\"\"
    logger.info("Fitting models on %d training documents...", len(train_docs))
    
    y = train_labels.map(label_to_idx).astype(int)
    
    # Fit BERTopic for interpretability
    topic_model.fit_transform(train_docs.tolist(), y=y.tolist())
    
    # Extract mpnet embeddings and train classifier
    X = extract_embeddings_mpnet(train_docs.tolist())
    clf.fit(X, y)
    
    logger.info("Model fitting complete.")

# Update predict_with_probs() function:
def predict_with_probs(topic_model, clf, texts, idx_to_label):
    \"\"\"Predict using mpnet embeddings and tuned SVM.\"\"\"
    X = extract_embeddings_mpnet(texts)
    probs = clf.predict_proba(X)
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [idx_to_label[int(i)] for i in pred_idx]
    return probs, pred_idx, pred_labels
""".format(
        classifier_path=BEST_CLASSIFIER_PATH,
        embedding_path=BEST_EMBEDDING_MODEL_PATH
    ))
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
