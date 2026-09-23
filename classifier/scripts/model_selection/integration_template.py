#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEMPLATE: Integration of Best Classifier into Main Script

Copy the relevant sections from this template into your
citymove_classification_optimized.py script.

This shows how to:
1. Load the saved classifier
2. Load the saved embedding model
3. Use them in your classification workflow
"""

import os
import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.svm import SVC

# =========================
# PATHS TO SAVED MODELS
# =========================

# Folder with the training and holdout workbooks (classifier/data in the published
# repository); set CITYMOVE_CLASSIFIER_DATA to use another folder.
DATA_DIR = os.environ.get(
    "CITYMOVE_CLASSIFIER_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "data"),
)

BEST_CLASSIFIER_PATH = os.path.join(DATA_DIR, "best_classifier.pkl")
BEST_EMBEDDING_MODEL_PATH = os.path.join(DATA_DIR, "best_embedding_model")

# =========================
# LOAD MODELS AT STARTUP
# =========================

print("="*80)
print("Loading best classifier and embedding model...")
print("="*80)

# Load pre-trained classifier
BEST_CLF = joblib.load(BEST_CLASSIFIER_PATH)
print(f"✓ Loaded classifier from: {BEST_CLASSIFIER_PATH}")

# Load embedding model
EMBEDDING_MODEL = SentenceTransformer(BEST_EMBEDDING_MODEL_PATH)
print(f"✓ Loaded embedding model from: {BEST_EMBEDDING_MODEL_PATH}")

print("="*80 + "\n")

# =========================
# HELPER FUNCTION
# =========================

def extract_embeddings_mpnet(docs):
    """
    Extract embeddings using all-mpnet-base-v2.
    
    Args:
        docs: List of document strings
        
    Returns:
        numpy array of shape (n_docs, 768)
    """
    return EMBEDDING_MODEL.encode(
        docs, 
        batch_size=32, 
        show_progress_bar=False
    )


# =========================
# UPDATED MODEL FUNCTIONS
# =========================

def build_models():
    """
    Build BERTopic (for interpretability) and use pre-trained SVM.
    
    Returns:
        topic_model: BERTopic for c-TF-IDF and visualizations
        clf: Pre-trained SVM classifier
    """
    # BERTopic for interpretability only
    empty_dimensionality_model = BaseDimensionalityReduction()
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    
    topic_model = BERTopic(
        umap_model=empty_dimensionality_model,
        ctfidf_model=ctfidf_model
    )
    
    # Use the pre-loaded best classifier
    # (This is already trained with optimal parameters)
    clf = BEST_CLF
    
    return topic_model, clf


def fit_everything(topic_model, clf, train_docs, train_labels, label_to_idx):
    """
    Fit both BERTopic (interpretability) and SVM (classification).
    
    Args:
        topic_model: BERTopic model
        clf: SVM classifier (will be retrained)
        train_docs: Training documents
        train_labels: Training labels
        label_to_idx: Label to index mapping
    """
    print(f"Fitting models on {len(train_docs)} training documents...")
    
    y = train_labels.map(label_to_idx).astype(int)
    
    # Fit BERTopic for interpretability (c-TF-IDF, top words)
    topic_model.fit_transform(train_docs.tolist(), y=y.tolist())
    
    # Extract mpnet embeddings
    print("Extracting embeddings...")
    X = extract_embeddings_mpnet(train_docs.tolist())
    
    # Train/retrain SVM classifier
    print("Training SVM...")
    clf.fit(X, y)
    
    print("✓ Model fitting complete")


def predict_with_probs(topic_model, clf, texts, idx_to_label):
    """
    Predict labels with confidence scores.
    
    Args:
        topic_model: BERTopic model (not used for prediction)
        clf: SVM classifier
        texts: List of texts to classify
        idx_to_label: Index to label mapping
        
    Returns:
        probs: Probability matrix (n_samples, n_classes)
        pred_idx: Predicted class indices
        pred_labels: Predicted label strings
    """
    # Extract embeddings
    X = extract_embeddings_mpnet(texts)
    
    # Predict
    probs = clf.predict_proba(X)
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [idx_to_label[int(i)] for i in pred_idx]
    
    return probs, pred_idx, pred_labels


# =========================
# EXAMPLE USAGE
# =========================

if __name__ == "__main__":
    # Example: Build models
    topic_model, clf = build_models()
    
    print("\nClassifier parameters:")
    params = clf.get_params()
    for key in ['kernel', 'C', 'gamma', 'class_weight']:
        if key in params:
            print(f"  {key:20s}: {params[key]}")
    
    # Example: Load some data and predict
    print("\nExample prediction:")
    
    test_texts = [
        "Air quality monitoring stations measuring PM2.5 and NO2 levels",
        "Bicycle lane infrastructure and cycling network coverage",
        "Population demographics and socioeconomic indicators"
    ]
    
    # Create dummy label mapping (normally you'd get this from your data)
    labels_unique = [
        'Active Transport Environment',
        'Air Quality',
        'Demographics & SES',
        'NCD\'s',
        'Other',
        'Recreational PA',
        'Recreational and Sports Environment',
        'Total PA',
        'Transport PA'
    ]
    idx_to_label = {i: lab for i, lab in enumerate(labels_unique)}
    
    # Predict
    probs, pred_idx, pred_labels = predict_with_probs(
        topic_model, clf, test_texts, idx_to_label
    )
    
    print("\nPredictions:")
    for i, text in enumerate(test_texts):
        print(f"\nText: {text[:60]}...")
        print(f"Predicted: {pred_labels[i]}")
        print(f"Confidence: {probs[i].max():.3f}")
    
    print("\n" + "="*80)
    print("Template complete! Copy relevant sections to your main script.")
    print("="*80)
