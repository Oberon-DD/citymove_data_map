#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classifier Comparison for CITY-MOVE Topic Classification

This script compares multiple classifiers on your existing labeled data
using the embeddings from BERTopic.

Steps:
1. Load your existing training/holdout data
2. Extract embeddings using BERTopic (or directly from sentence-transformers)
3. Compare 8 different classifiers
4. Show detailed performance metrics
5. Identify the best classifier
6. Save results for hyperparameter tuning

Usage:
    python compare_classifiers.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support
)

# Try to import XGBoost (not always installed)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost not installed. Install with: pip install xgboost --break-system-packages")

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠ sentence-transformers not installed (embeddings will use BERTopic)")

# =========================
# CONFIG
# =========================

# Update these paths to match your setup
# Folder with the training and holdout workbooks (classifier/data in the published
# repository); set CITYMOVE_CLASSIFIER_DATA to use another folder.
DATA_DIR = os.environ.get(
    "CITYMOVE_CLASSIFIER_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "data"),
)

TOPIC_MODEL_PATH = os.path.join(DATA_DIR, "bertopic_model.pkl")
TRAIN_DOCS_PATH = os.path.join(DATA_DIR, "train_docs.xlsx")
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "train_labels.xlsx")
HOLDOUT_DOCS_PATH = os.path.join(DATA_DIR, "holdout_docs.xlsx")
HOLDOUT_LABELS_PATH = os.path.join(DATA_DIR, "holdout_labels.xlsx")

OUTPUT_DIR = DATA_DIR
RESULTS_FILE = os.path.join(OUTPUT_DIR, "classifier_comparison_results.xlsx")
PLOT_FILE = os.path.join(OUTPUT_DIR, "classifier_comparison.png")

# Random seed for reproducibility
RANDOM_STATE = 42

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
# DATA LOADING
# =========================

def load_data() -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load training and holdout data."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Check files exist
    required_files = [TRAIN_DOCS_PATH, TRAIN_LABELS_PATH, HOLDOUT_DOCS_PATH, HOLDOUT_LABELS_PATH]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"\n❌ ERROR: Missing files:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)
    
    # Load data
    train_docs = pd.read_excel(TRAIN_DOCS_PATH).iloc[:, 0].astype(str)
    train_labels = pd.read_excel(TRAIN_LABELS_PATH).iloc[:, 0].astype(str).map(normalize_label)
    holdout_docs = pd.read_excel(HOLDOUT_DOCS_PATH).iloc[:, 0].astype(str)
    holdout_labels = pd.read_excel(HOLDOUT_LABELS_PATH).iloc[:, 0].astype(str).map(normalize_label)
    
    print(f"\n✓ Training set:  {len(train_docs):4d} documents")
    print(f"✓ Holdout set:   {len(holdout_docs):4d} documents")
    print(f"\n✓ Unique labels: {len(train_labels.unique())}")
    
    # Show class distribution
    print("\nTraining set class distribution:")
    class_counts = train_labels.value_counts().sort_index()
    for label, count in class_counts.items():
        print(f"  {label:40s}: {count:4d} ({count/len(train_labels)*100:5.1f}%)")
    
    return train_docs, train_labels, holdout_docs, holdout_labels


# =========================
# EMBEDDING EXTRACTION
# =========================

def extract_embeddings(
    train_docs: pd.Series,
    holdout_docs: pd.Series,
    method: str = 'bertopic'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings using BERTopic or directly from sentence-transformers.
    
    Args:
        method: 'bertopic' or 'sentence-transformers'
    """
    print("\n" + "="*80)
    print(f"EXTRACTING EMBEDDINGS (method={method})")
    print("="*80)
    
    if method == 'bertopic':
        if not os.path.exists(TOPIC_MODEL_PATH):
            print(f"\n❌ ERROR: BERTopic model not found at {TOPIC_MODEL_PATH}")
            print("   Please train the model first or use method='sentence-transformers'")
            sys.exit(1)
        
        print(f"\nLoading BERTopic model from: {TOPIC_MODEL_PATH}")
        topic_model = joblib.load(TOPIC_MODEL_PATH)
        
        print("Extracting embeddings...")
        X_train = topic_model._extract_embeddings(train_docs.tolist())
        X_holdout = topic_model._extract_embeddings(holdout_docs.tolist())
        
    elif method == 'sentence-transformers':
        if not HAS_SENTENCE_TRANSFORMERS:
            print("\n❌ ERROR: sentence-transformers not installed")
            print("   Install with: pip install sentence-transformers --break-system-packages")
            sys.exit(1)
        
        print("\nLoading sentence-transformers model: all-MiniLM-L6-v2")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Extracting embeddings...")
        X_train = model.encode(train_docs.tolist(), show_progress_bar=True)
        X_holdout = model.encode(holdout_docs.tolist(), show_progress_bar=True)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"\n✓ Training embeddings:  {X_train.shape}")
    print(f"✓ Holdout embeddings:   {X_holdout.shape}")
    
    return X_train, X_holdout


# =========================
# CLASSIFIER DEFINITIONS
# =========================

def get_classifiers() -> Dict[str, object]:
    """
    Define classifiers to compare.
    
    Returns dict with classifier name -> classifier object.
    """
    classifiers = {
        # SVM variants
        'SVM (RBF, default)': SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=RANDOM_STATE
        ),
        
        'SVM (RBF, balanced)': SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        
        'SVM (Linear, balanced)': SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        
        # Logistic Regression
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        
        # Random Forest
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        
        # Gradient Boosting
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE
        ),
        
        # K-Nearest Neighbors
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        ),
    }
    
    # Add XGBoost if available
    if HAS_XGBOOST:
        classifiers['XGBoost'] = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss'
        )
    
    return classifiers


# =========================
# EVALUATION
# =========================

def evaluate_classifier(
    clf,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_holdout: np.ndarray,
    y_holdout: pd.Series,
    name: str
) -> Dict:
    """
    Train and evaluate a single classifier.
    
    Returns dict with performance metrics.
    """
    # Get label mappings
    labels_unique = sorted(y_train.unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels_unique)}
    
    # Convert labels to indices
    y_train_idx = y_train.map(label_to_idx).values
    y_holdout_idx = y_holdout.map(label_to_idx).values
    
    # Train
    print(f"\n  Training {name}...")
    clf.fit(X_train, y_train_idx)
    
    # Predict
    y_pred = clf.predict(X_holdout)
    
    # Metrics
    accuracy = accuracy_score(y_holdout_idx, y_pred)
    f1_weighted = f1_score(y_holdout_idx, y_pred, average='weighted')
    f1_macro = f1_score(y_holdout_idx, y_pred, average='macro')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_holdout_idx, y_pred, average=None, labels=range(len(labels_unique))
    )
    
    results = {
        'name': name,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'classifier': clf,
        'y_pred': y_pred,
        'per_class': {
            'labels': labels_unique,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    }
    
    print(f"    Accuracy:    {accuracy:.4f}")
    print(f"    F1 (weighted): {f1_weighted:.4f}")
    print(f"    F1 (macro):    {f1_macro:.4f}")
    
    return results


def compare_all_classifiers(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_holdout: np.ndarray,
    y_holdout: pd.Series
) -> pd.DataFrame:
    """
    Compare all classifiers and return results DataFrame.
    """
    print("\n" + "="*80)
    print("COMPARING CLASSIFIERS")
    print("="*80)
    
    classifiers = get_classifiers()
    all_results = []
    
    for name, clf in classifiers.items():
        results = evaluate_classifier(clf, X_train, y_train, X_holdout, y_holdout, name)
        all_results.append(results)
    
    # Create summary DataFrame
    summary = pd.DataFrame([
        {
            'Classifier': r['name'],
            'Accuracy': r['accuracy'],
            'F1 (Weighted)': r['f1_weighted'],
            'F1 (Macro)': r['f1_macro']
        }
        for r in all_results
    ])
    
    # Sort by F1 weighted (descending)
    summary = summary.sort_values('F1 (Weighted)', ascending=False).reset_index(drop=True)
    
    return summary, all_results


# =========================
# VISUALIZATION
# =========================

def plot_comparison(summary: pd.DataFrame, output_path: str):
    """Create comparison visualization."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Sort by F1 weighted for consistent ordering
    summary_sorted = summary.sort_values('F1 (Weighted)', ascending=True)
    
    # Plot 1: Accuracy
    axes[0].barh(summary_sorted['Classifier'], summary_sorted['Accuracy'], color='steelblue')
    axes[0].set_xlabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: F1 Weighted
    axes[1].barh(summary_sorted['Classifier'], summary_sorted['F1 (Weighted)'], color='coral')
    axes[1].set_xlabel('F1 Score (Weighted)', fontsize=12)
    axes[1].set_title('F1 Weighted Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].grid(axis='x', alpha=0.3)
    
    # Plot 3: F1 Macro
    axes[2].barh(summary_sorted['Classifier'], summary_sorted['F1 (Macro)'], color='mediumseagreen')
    axes[2].set_xlabel('F1 Score (Macro)', fontsize=12)
    axes[2].set_title('F1 Macro Comparison', fontsize=14, fontweight='bold')
    axes[2].set_xlim(0, 1)
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    plt.show()


def plot_best_confusion_matrix(
    best_result: Dict,
    y_holdout: pd.Series,
    output_path: str
):
    """Plot confusion matrix for best classifier."""
    labels_unique = sorted(y_holdout.unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels_unique)}
    
    y_true = y_holdout.map(label_to_idx).values
    y_pred = best_result['y_pred']
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels_unique)))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels_unique,
        yticklabels=labels_unique,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f"Confusion Matrix - {best_result['name']}", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = output_path.replace('.png', '_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to: {cm_path}")
    plt.show()


# =========================
# DETAILED REPORTING
# =========================

def print_detailed_results(all_results: List[Dict], y_holdout: pd.Series):
    """Print detailed per-class results for best classifier."""
    print("\n" + "="*80)
    print("DETAILED RESULTS FOR BEST CLASSIFIER")
    print("="*80)
    
    # Find best by F1 weighted
    best = max(all_results, key=lambda x: x['f1_weighted'])
    
    print(f"\nBest Classifier: {best['name']}")
    print(f"Overall Accuracy:    {best['accuracy']:.4f}")
    print(f"F1 (Weighted):       {best['f1_weighted']:.4f}")
    print(f"F1 (Macro):          {best['f1_macro']:.4f}")
    
    print("\nPer-Class Performance:")
    print("-" * 80)
    print(f"{'Label':<40} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 80)
    
    per_class = best['per_class']
    for i, label in enumerate(per_class['labels']):
        print(
            f"{label:<40} "
            f"{per_class['precision'][i]:>10.3f} "
            f"{per_class['recall'][i]:>10.3f} "
            f"{per_class['f1'][i]:>10.3f} "
            f"{int(per_class['support'][i]):>10d}"
        )
    
    print("-" * 80)
    
    # Get full classification report
    labels_unique = sorted(y_holdout.unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels_unique)}
    y_true = y_holdout.map(label_to_idx).values
    y_pred = best['y_pred']
    
    print("\nFull Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels_unique, digits=3))
    
    return best


# =========================
# EXPORT RESULTS
# =========================

def export_results(summary: pd.DataFrame, all_results: List[Dict], output_path: str):
    """Export detailed results to Excel."""
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Summary
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Per-class results for each classifier
        per_class_data = []
        for result in all_results:
            pc = result['per_class']
            for i, label in enumerate(pc['labels']):
                per_class_data.append({
                    'Classifier': result['name'],
                    'Label': label,
                    'Precision': pc['precision'][i],
                    'Recall': pc['recall'][i],
                    'F1': pc['f1'][i],
                    'Support': int(pc['support'][i])
                })
        
        per_class_df = pd.DataFrame(per_class_data)
        per_class_df.to_excel(writer, sheet_name='Per-Class Metrics', index=False)
    
    print(f"\n✓ Saved detailed results to: {output_path}")


# =========================
# MAIN
# =========================

def main():
    """Main comparison workflow."""
    print("\n" + "="*80)
    print("CLASSIFIER COMPARISON FOR CITY-MOVE TOPIC CLASSIFICATION")
    print("="*80)
    
    # Step 1: Load data
    train_docs, train_labels, holdout_docs, holdout_labels = load_data()
    
    # Step 2: Extract embeddings
    # Options: 'bertopic' (uses your existing model) or 'sentence-transformers' (direct)
    X_train, X_holdout = extract_embeddings(
        train_docs, 
        holdout_docs,
        method='bertopic'  # Change to 'sentence-transformers' if you prefer
    )
    
    # Step 3: Compare classifiers
    summary, all_results = compare_all_classifiers(
        X_train, train_labels,
        X_holdout, holdout_labels
    )
    
    # Step 4: Display results
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    print("\n" + summary.to_string(index=False))
    
    # Step 5: Detailed results for best classifier
    best_result = print_detailed_results(all_results, holdout_labels)
    
    # Step 6: Visualizations
    plot_comparison(summary, PLOT_FILE)
    plot_best_confusion_matrix(best_result, holdout_labels, PLOT_FILE)
    
    # Step 7: Export
    export_results(summary, all_results, RESULTS_FILE)
    
    # Step 8: Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    best_name = summary.iloc[0]['Classifier']
    best_f1 = summary.iloc[0]['F1 (Weighted)']
    
    print(f"\n✓ Best classifier: {best_name}")
    print(f"✓ F1 Score (Weighted): {best_f1:.4f}")
    
    print("\nNext steps:")
    print("1. Review the confusion matrix to identify problem classes")
    print("2. Run hyperparameter tuning on the best classifier")
    print("3. Consider ensemble methods if multiple classifiers perform similarly")
    
    if best_f1 < 0.85:
        print("\n⚠ F1 score is below 0.85. Consider:")
        print("  - Using better embeddings (all-mpnet-base-v2)")
        print("  - More training data (continue active learning)")
        print("  - Trying SetFit for few-shot learning")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
