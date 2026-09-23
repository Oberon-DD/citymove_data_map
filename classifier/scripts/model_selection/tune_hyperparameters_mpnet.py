#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Tuning for CITY-MOVE Classification

This script tunes hyperparameters for the best-performing classifier
using all-mpnet-base-v2 embeddings (best from embedding comparison).

After running compare_classifiers.py and test_embeddings.py, use this
script to find optimal parameters.

Usage:
    python tune_hyperparameters.py --classifier svm
    python tune_hyperparameters.py --classifier logistic
    python tune_hyperparameters.py --classifier xgboost
    python tune_hyperparameters.py --classifier random_forest
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, classification_report
from sentence_transformers import SentenceTransformer

# Try to import XGBoost (not always installed)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost not installed. Install with: pip install xgboost --break-system-packages")

# =========================
# CONFIG
# =========================

# Folder with the training and holdout workbooks (classifier/data in the published
# repository); set CITYMOVE_CLASSIFIER_DATA to use another folder.
DATA_DIR = os.environ.get(
    "CITYMOVE_CLASSIFIER_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "data"),
)

TRAIN_DOCS_PATH = os.path.join(DATA_DIR, "train_docs.xlsx")
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "train_labels.xlsx")

OUTPUT_DIR = DATA_DIR
TUNED_MODEL_PATH = os.path.join(OUTPUT_DIR, "tuned_classifier.pkl")
TUNING_RESULTS_PATH = os.path.join(OUTPUT_DIR, "hyperparameter_tuning_results.xlsx")

# Use best embedding model from test_embeddings.py results
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'  # Winner: +1.8% F1 weighted, +5.1% F1 macro

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
# PARAMETER GRIDS
# =========================

def get_param_grid(classifier_type: str) -> Dict[str, Any]:
    """
    Get parameter grid for specified classifier.
    
    These grids are designed for your specific use case:
    - Short text (median 93 chars)
    - Imbalanced classes
    - Small-to-medium training set
    - Using all-mpnet-base-v2 embeddings (768 dimensions)
    """
    
    if classifier_type == 'svm':
        # Comprehensive grid for SVM
        return {
            'kernel': ['rbf', 'linear'],
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'class_weight': ['balanced', None]
        }
    
    elif classifier_type == 'svm_fast':
        # Faster grid (less combinations)
        return {
            'kernel': ['rbf'],
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 0.01, 0.1],
            'class_weight': ['balanced']
        }
    
    elif classifier_type == 'logistic':
        return {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['saga'],  # saga supports all penalties
            'class_weight': ['balanced', None],
            'max_iter': [1000, 2000]
        }
    
    elif classifier_type == 'random_forest':
        return {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
    
    elif classifier_type == 'gradient_boosting':
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 1.0]
        }
    
    elif classifier_type == 'xgboost':
        if not HAS_XGBOOST:
            raise ValueError("XGBoost not installed")
        
        return {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
    
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def get_base_classifier(classifier_type: str):
    """Get base classifier object."""
    
    if classifier_type in ['svm', 'svm_fast']:
        return SVC(probability=True, random_state=RANDOM_STATE)
    
    elif classifier_type == 'logistic':
        return LogisticRegression(random_state=RANDOM_STATE)
    
    elif classifier_type == 'random_forest':
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    elif classifier_type == 'gradient_boosting':
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    
    elif classifier_type == 'xgboost':
        if not HAS_XGBOOST:
            raise ValueError("XGBoost not installed")
        return XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss')
    
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


# =========================
# DATA LOADING
# =========================

def load_data():
    """Load training data and extract embeddings using all-mpnet-base-v2."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load text and labels
    train_docs = pd.read_excel(TRAIN_DOCS_PATH).iloc[:, 0].astype(str)
    train_labels = pd.read_excel(TRAIN_LABELS_PATH).iloc[:, 0].astype(str).map(normalize_label)
    
    print(f"\n✓ Training set: {len(train_docs)} documents")
    print(f"✓ Unique labels: {len(train_labels.unique())}")
    
    # Show class distribution
    print("\nClass distribution:")
    for label, count in train_labels.value_counts().sort_index().items():
        print(f"  {label:40s}: {count:4d} ({count/len(train_labels)*100:5.1f}%)")
    
    # Extract embeddings using all-mpnet-base-v2
    print("\n" + "="*80)
    print(f"EXTRACTING EMBEDDINGS: {EMBEDDING_MODEL_NAME}")
    print("="*80)
    
    print(f"\nLoading {EMBEDDING_MODEL_NAME}...")
    print("(This is the best-performing model from test_embeddings.py)")
    print("768-dimensional embeddings, ~420MB download on first run")
    
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print(f"\nExtracting embeddings for {len(train_docs)} documents...")
    X = embedding_model.encode(
        train_docs.tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    
    print(f"\n✓ Embeddings shape: {X.shape}")
    print(f"  Dimensions: {X.shape[1]} (768-dim mpnet)")
    
    # Convert labels to indices
    labels_unique = sorted(train_labels.unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels_unique)}
    y = train_labels.map(label_to_idx).values
    
    return X, y, labels_unique


# =========================
# HYPERPARAMETER TUNING
# =========================

def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    classifier_type: str,
    cv_folds: int = 5,
    n_jobs: int = -1
):
    """
    Perform grid search for hyperparameter tuning.
    
    Args:
        X: Training embeddings (from all-mpnet-base-v2)
        y: Training labels (as indices)
        classifier_type: Type of classifier to tune
        cv_folds: Number of cross-validation folds
        n_jobs: Number of parallel jobs (-1 = use all cores)
    """
    print("\n" + "="*80)
    print(f"HYPERPARAMETER TUNING: {classifier_type.upper()}")
    print("="*80)
    
    # Get classifier and parameter grid
    base_clf = get_base_classifier(classifier_type)
    param_grid = get_param_grid(classifier_type)
    
    # Calculate total combinations
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)
    
    print(f"\nBase classifier: {base_clf.__class__.__name__}")
    print(f"\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param:20s}: {values}")
    
    print(f"\nTotal combinations to test: {total_combinations}")
    print(f"Cross-validation folds: {cv_folds}")
    print(f"Total model fits: {total_combinations * cv_folds}")
    
    # Estimate time
    if classifier_type in ['svm', 'svm_fast']:
        time_per_fit = 0.5  # seconds
    else:
        time_per_fit = 0.2
    
    estimated_time = (total_combinations * cv_folds * time_per_fit) / 60
    print(f"Estimated time: {estimated_time:.1f} minutes")
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # Set up scoring
    scorer = make_scorer(f1_score, average='weighted')
    
    # Grid search
    print(f"\nStarting grid search...")
    print("=" * 80)
    
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True
    )
    
    grid_search.fit(X, y)
    
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    
    print(f"\nBest parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param:20s}: {value}")
    
    print(f"\nBest cross-validation F1 score (weighted): {grid_search.best_score_:.4f}")
    
    # Get standard deviation of best model
    best_index = grid_search.best_index_
    best_std = grid_search.cv_results_['std_test_score'][best_index]
    print(f"Standard deviation: ±{best_std:.4f}")
    
    return grid_search


# =========================
# RESULTS ANALYSIS
# =========================

def analyze_results(grid_search, labels_unique: list):
    """Analyze and visualize grid search results."""
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    
    # Get results DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Sort by mean test score
    results_df = results_df.sort_values('mean_test_score', ascending=False)
    
    # Top 10 parameter combinations
    print("\nTop 10 parameter combinations:")
    print("-" * 80)
    
    top_10 = results_df.head(10)[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    for idx, row in top_10.iterrows():
        print(f"\nRank {int(row['rank_test_score'])}:")
        print(f"  Mean F1: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        print(f"  Params: {row['params']}")
    
    # Check for overfitting
    print("\n" + "-" * 80)
    print("Overfitting Analysis (Top 5 models):")
    print("-" * 80)
    
    for idx, row in results_df.head(5).iterrows():
        train_score = row['mean_train_score']
        test_score = row['mean_test_score']
        gap = train_score - test_score
        
        print(f"\nRank {int(row['rank_test_score'])}:")
        print(f"  Train F1: {train_score:.4f}")
        print(f"  Test F1:  {test_score:.4f}")
        print(f"  Gap:      {gap:.4f}", end="")
        
        if gap > 0.1:
            print(" ⚠ High overfitting")
        elif gap > 0.05:
            print(" ⚠ Moderate overfitting")
        else:
            print(" ✓ Good generalization")
    
    return results_df


def plot_tuning_results(results_df: pd.DataFrame, classifier_type: str):
    """Visualize hyperparameter tuning results."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Score distribution
    axes[0, 0].hist(results_df['mean_test_score'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(
        results_df['mean_test_score'].max(),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Best: {results_df['mean_test_score'].max():.4f}"
    )
    axes[0, 0].set_xlabel('Mean Test Score (F1 Weighted)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Distribution of Cross-Validation Scores', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Top 20 configurations
    top_20 = results_df.head(20).copy()
    top_20 = top_20.sort_values('mean_test_score', ascending=True)
    
    axes[0, 1].barh(range(len(top_20)), top_20['mean_test_score'], color='steelblue')
    axes[0, 1].errorbar(
        top_20['mean_test_score'],
        range(len(top_20)),
        xerr=top_20['std_test_score'],
        fmt='none',
        color='black',
        alpha=0.5
    )
    axes[0, 1].set_xlabel('Mean Test Score (F1 Weighted)', fontsize=12)
    axes[0, 1].set_ylabel('Configuration Rank', fontsize=12)
    axes[0, 1].set_title('Top 20 Configurations (with std dev)', fontsize=14, fontweight='bold')
    axes[0, 1].set_yticks(range(len(top_20)))
    axes[0, 1].set_yticklabels([f"#{i+1}" for i in range(len(top_20))])
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Plot 3: Train vs Test (overfitting check)
    top_50 = results_df.head(50)
    axes[1, 0].scatter(top_50['mean_train_score'], top_50['mean_test_score'], alpha=0.6, s=50)
    
    # Add diagonal line (perfect generalization)
    min_score = min(top_50['mean_train_score'].min(), top_50['mean_test_score'].min())
    max_score = max(top_50['mean_train_score'].max(), top_50['mean_test_score'].max())
    axes[1, 0].plot([min_score, max_score], [min_score, max_score], 'r--', label='Perfect generalization')
    
    axes[1, 0].set_xlabel('Train Score', fontsize=12)
    axes[1, 0].set_ylabel('Test Score', fontsize=12)
    axes[1, 0].set_title('Train vs Test Performance (Top 50)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Ranking stability (std dev)
    top_30 = results_df.head(30).copy()
    top_30 = top_30.sort_values('std_test_score', ascending=True)
    
    colors = ['green' if std < 0.02 else 'orange' if std < 0.04 else 'red' 
              for std in top_30['std_test_score']]
    
    axes[1, 1].barh(range(len(top_30)), top_30['std_test_score'], color=colors)
    axes[1, 1].set_xlabel('Standard Deviation of Test Score', fontsize=12)
    axes[1, 1].set_ylabel('Configuration', fontsize=12)
    axes[1, 1].set_title('Model Stability (Top 30, lower is better)', fontsize=14, fontweight='bold')
    axes[1, 1].set_yticks(range(len(top_30)))
    axes[1, 1].set_yticklabels([f"#{i+1}" for i in range(len(top_30))])
    axes[1, 1].axvline(0.02, color='green', linestyle='--', alpha=0.5, label='Good (<0.02)')
    axes[1, 1].axvline(0.04, color='orange', linestyle='--', alpha=0.5, label='Moderate (<0.04)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, f'{classifier_type}_tuning_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {plot_path}")
    plt.show()


# =========================
# EXPORT
# =========================

def save_results(grid_search, results_df: pd.DataFrame, classifier_type: str):
    """Save tuned model and detailed results."""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save best model
    model_path = TUNED_MODEL_PATH.replace('.pkl', f'_{classifier_type}.pkl')
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"\n✓ Saved tuned model to: {model_path}")
    
    # Save detailed results
    results_path = TUNING_RESULTS_PATH.replace('.xlsx', f'_{classifier_type}.xlsx')
    
    with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
        # Sheet 1: Best parameters
        best_params_df = pd.DataFrame([
            {'Parameter': k, 'Value': v}
            for k, v in grid_search.best_params_.items()
        ])
        best_params_df.to_excel(writer, sheet_name='Best Parameters', index=False)
        
        # Sheet 2: Summary stats
        summary = pd.DataFrame([{
            'Best F1 Score': grid_search.best_score_,
            'Best Std Dev': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
            'Total Combinations Tested': len(results_df),
            'Embedding Model': EMBEDDING_MODEL_NAME,
            'Embedding Dimensions': 768,
            'Random State': RANDOM_STATE
        }])
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 3: Top 50 configurations
        top_configs = results_df.head(50)[
            ['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 
             'std_train_score', 'rank_test_score']
        ].copy()
        top_configs.to_excel(writer, sheet_name='Top 50 Configs', index=False)
        
        # Sheet 4: All results
        results_df.to_excel(writer, sheet_name='All Results', index=False)
    
    print(f"✓ Saved detailed results to: {results_path}")
    
    return model_path


# =========================
# MAIN
# =========================

def main():
    """Main tuning workflow."""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for CITY-MOVE classifier')
    parser.add_argument(
        '--classifier',
        type=str,
        required=True,
        choices=['svm', 'svm_fast', 'logistic', 'random_forest', 'gradient_boosting', 'xgboost'],
        help='Classifier type to tune'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (default: -1, use all cores)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING FOR CITY-MOVE CLASSIFICATION")
    print("="*80)
    print(f"\nClassifier: {args.classifier}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"CV Folds: {args.cv_folds}")
    print(f"Parallel Jobs: {args.n_jobs}")
    
    # Load data and extract embeddings
    X, y, labels_unique = load_data()
    
    # Tune hyperparameters
    grid_search = tune_hyperparameters(
        X, y,
        classifier_type=args.classifier,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs
    )
    
    # Analyze results
    results_df = analyze_results(grid_search, labels_unique)
    
    # Visualize
    plot_tuning_results(results_df, args.classifier)
    
    # Save
    model_path = save_results(grid_search, results_df, args.classifier)
    
    # Final recommendations
    print("\n" + "="*80)
    print("TUNING COMPLETE!")
    print("="*80)
    
    print(f"\nBest F1 score: {grid_search.best_score_:.4f}")
    
    print(f"\nOptimal parameters to use in your main script:")
    print("-" * 80)
    for param, value in grid_search.best_params_.items():
        print(f"  {param:20s} = {value}")
    
    print("\n" + "-" * 80)
    print("Integration example for your main script:")
    print("-" * 80)
    print(f"""
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC

# Load embedding model
embedding_model = SentenceTransformer('{EMBEDDING_MODEL_NAME}')

# Extract embeddings
def extract_embeddings(docs):
    return embedding_model.encode(docs, batch_size=32, show_progress_bar=False)

# Use tuned classifier
clf = SVC(
""")
    for param, value in grid_search.best_params_.items():
        if isinstance(value, str):
            print(f"    {param}='{value}',")
        else:
            print(f"    {param}={value},")
    print(f"    probability=True,")
    print(f"    random_state={RANDOM_STATE}")
    print(")")
    
    print("\nNext steps:")
    print("1. Update your main classification script with the optimal parameters")
    print("2. Update embedding extraction to use all-mpnet-base-v2")
    print("3. Re-run evaluation on holdout set to confirm performance")
    print("4. Continue active learning with the tuned classifier")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
