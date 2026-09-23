#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CITY-MOVE | Inference, Diagnostics & Prediction Validation
Version: 4.0

Combined script merging interpretation/diagnostics and prediction validation.
Runs on docs_and_labels_augmented.xlsx + to_predict_df.xlsx.

Outputs
-------
A. Inference & Diagnostics:
   - Full predictions with confidence, margin, entropy
   - City × label breakdown (counts + percentages)
   - Confidence threshold analysis
   - Confidence by city, by label, by city×label
   - Confusion analysis (top-1 vs top-2 prediction)
   - Per-label TF-IDF word extraction (top words + discriminative words)
   - UMAP / t-SNE embedding visualisation
   - Centroid distance heatmap

B. Prediction Validation:
   - Holdout evaluation (static + dynamic) with classification reports
   - Calibration reliability diagrams
   - Threshold sweep tables (per city, per city×label)
   - City similarity to training distribution (OOD diagnostics)
   - OOD scatter: distance to label centroid vs confidence
   - Diagnostics summary

8 classes: Total PA and Recreational PA kept as separate categories.

Author: Oberon
Date: 2026-03
"""

from __future__ import annotations

import os
import platform
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

import joblib
from sentence_transformers import SentenceTransformer


# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    sns = None
    _HAS_SEABORN = False

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    nx = None
    _HAS_NETWORKX = False

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    umap = None
    _HAS_UMAP = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# Folder with classifier.pkl and the training, holdout and target workbooks.
# Defaults to classifier/data in the published repository; set
# CITYMOVE_CLASSIFIER_DATA to use another folder.
DATA_DIR = os.environ.get(
    "CITYMOVE_CLASSIFIER_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"),
)

OUTPUTS_DIR = os.path.join(DATA_DIR, "DIAGNOSTICS_OUTPUTS")

# ── Input files ─────────────────────────────────────────────────────────────
AUGMENTED_DATA_PATH = os.path.join(DATA_DIR, "docs_and_labels_augmented.xlsx")
TARGET_FILE = os.path.join(DATA_DIR, "to_predict_df.xlsx")
CLF_PATH = os.path.join(DATA_DIR, "classifier.pkl")
LABEL_MAPS_PATH = os.path.join(DATA_DIR, "label_maps.pkl")

# ── Optional holdouts ───────────────────────────────────────────────────────
STATIC_HOLDOUT_PATH = os.path.join(DATA_DIR, "holdout_static.xlsx")
DYNAMIC_HOLDOUT_PATH = os.path.join(DATA_DIR, "holdout_dynamic.xlsx")

# ── Column names ────────────────────────────────────────────────────────────
DOC_COL = "docs"            # column in training / holdout data
LABEL_COL = "labels"        # column in training / holdout data
TEXT_COL = "doc"             # column in target file
IS_SYNTHETIC_COL = "is_synthetic"

# ── Embedding model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_BATCH_SIZE = 64

# ── City mapping ────────────────────────────────────────────────────────────
CITY_MAPPING = {
    "df_a": "Ljubljana",
    "df_b": "Kampala",
    "df_c": "Antwerp",
    "df_d": "Bogota",
    "df_e": "Lima",
    "df_f": "Rotterdam",
    "df_g": "Antwerp",
    "df_h": "Bogota",
    "df_i": "Rotterdam",
    "df_j": "Bogota",
}

# ── Visualization parameters ────────────────────────────────────────────────
DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CONFUSION_NETWORK_MIN_COUNT = 50
EMBED_VIZ_N_PER_CLASS = 300
EMBED_VIZ_REDUCER = "umap"   # "umap" or "tsne"
STORE_FULL_PROBS_IN_DF = False
EXCLUDE_LABELS_IN_VIZ = {"Other"}
CALIBRATION_N_BINS = 10
DPI = 300

# ── Color maps ──────────────────────────────────────────────────────────────
CMAP_BLUE_YELLOW = "YlGnBu"
CMAP_TURQUOISE = "YlGn"
CMAP_TURQUOISE_REV = "GnBu"


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("citymove_diagnostics")


# =============================================================================
# LABEL NORMALIZATION
# =============================================================================

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
}


def normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return label
    return CANONICAL_MAP.get(label.strip().lower(), label.strip())


# =============================================================================
# HELPERS
# =============================================================================

def _filter_excluded(df, col="Predicted_label", exclude=EXCLUDE_LABELS_IN_VIZ):
    if col not in df.columns:
        return df
    return df[~df[col].isin(exclude)].copy()


def _labels_without_excluded(labels, exclude=EXCLUDE_LABELS_IN_VIZ):
    return [lab for lab in labels if lab not in exclude]


def _softmax(z, axis=1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def prediction_entropy(probs, eps=1e-12):
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def encode_texts(emb_model, texts, batch_size=EMBED_BATCH_SIZE):
    return np.asarray(emb_model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    ))


def get_probabilities(clf, X):
    if hasattr(clf, "predict_proba"):
        return np.asarray(clf.predict_proba(X))
    if hasattr(clf, "decision_function"):
        scores = np.asarray(clf.decision_function(X))
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        return _softmax(scores, axis=1)
    raise TypeError(f"Classifier must provide predict_proba() or decision_function().")


# =============================================================================
# DATA CONTAINER
# =============================================================================

@dataclass
class LoadedAssets:
    """All loaded artifacts needed for inference + validation."""
    clf: object
    emb_model: SentenceTransformer
    train_df: pd.DataFrame
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    labels_unique: List[str]
    label_words: Optional[Dict[str, List[Tuple[str, float]]]] = None
    X_train: Optional[np.ndarray] = None
    train_centroid: Optional[np.ndarray] = None
    label_centroids: Dict[str, np.ndarray] = field(default_factory=dict)


# =============================================================================
# TF-IDF WORD EXTRACTION
# =============================================================================

def build_label_words(train_df, labels_unique, max_features=5000, top_n=30):
    """Extract top words per label using direct TF-IDF on label-grouped docs."""
    logger.info("Computing per-label TF-IDF words\u2026")
    class_docs = {}
    for lab in labels_unique:
        texts = train_df[train_df[LABEL_COL] == lab][DOC_COL].tolist()
        class_docs[lab] = " ".join(str(t) for t in texts)

    ordered_labels = list(class_docs.keys())
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = tfidf.fit_transform([class_docs[lab] for lab in ordered_labels])
    vocab = tfidf.get_feature_names_out()

    label_words = {}
    for i, lab in enumerate(ordered_labels):
        scores = X[i].toarray().flatten()
        top_idx = scores.argsort()[::-1][:top_n]
        label_words[lab] = [(vocab[j], float(scores[j])) for j in top_idx if scores[j] > 0]

    logger.info("\u2713 Extracted top words for %d labels", len(label_words))
    return label_words


# =============================================================================
# LOAD DATA & MODELS
# =============================================================================

def _load_classifier():
    """Load pickled SVM classifier."""
    if not os.path.exists(CLF_PATH):
        raise FileNotFoundError(f"Classifier not found: {CLF_PATH}")
    clf = joblib.load(CLF_PATH)
    logger.info("\u2713 Loaded classifier")
    return clf


def _load_training_data():
    """Load and normalise augmented training data."""
    if not os.path.exists(AUGMENTED_DATA_PATH):
        raise FileNotFoundError(f"Training data not found: {AUGMENTED_DATA_PATH}")
    train_df = pd.read_excel(AUGMENTED_DATA_PATH)
    if DOC_COL not in train_df.columns or LABEL_COL not in train_df.columns:
        raise ValueError(f"Expected '{DOC_COL}' and '{LABEL_COL}'. Found: {list(train_df.columns)}")
    train_df[DOC_COL] = train_df[DOC_COL].astype(str).fillna("")
    train_df[LABEL_COL] = train_df[LABEL_COL].astype(str).map(normalize_label)
    logger.info("\u2713 Loaded training data: %d docs", len(train_df))
    return train_df


def _load_label_maps(train_df):
    """Load label maps from pkl or rebuild from data."""
    if os.path.exists(LABEL_MAPS_PATH):
        maps = joblib.load(LABEL_MAPS_PATH)
        l2i = maps["label_to_idx"]
        i2l = {int(k): v for k, v in maps["idx_to_label"].items()}
        labels = maps["labels_unique"]
        logger.info("\u2713 Label maps from pkl: %d labels", len(labels))
    else:
        logger.warning("label_maps.pkl not found \u2014 rebuilding from data.")
        labels = sorted(train_df[LABEL_COL].unique().tolist())
        l2i = {lab: i for i, lab in enumerate(labels)}
        i2l = {i: lab for lab, i in l2i.items()}
    return l2i, i2l, labels


def _compute_centroids(emb_model, train_df, labels_unique):
    """Encode training data and compute global + per-label centroids."""
    logger.info("Encoding training embeddings for centroids\u2026")
    X_train = encode_texts(emb_model, train_df[DOC_COL].tolist())
    train_centroid = X_train.mean(axis=0)
    label_centroids = {}
    for lab in labels_unique:
        mask = train_df[LABEL_COL].values == lab
        if mask.sum() >= 2:
            label_centroids[lab] = X_train[mask].mean(axis=0)
    logger.info("\u2713 Centroids: %d / %d labels", len(label_centroids), len(labels_unique))
    return X_train, train_centroid, label_centroids


def load_assets(compute_centroids=True):
    """Load classifier, training data, embedding model, label maps, centroids."""
    clf = _load_classifier()
    train_df = _load_training_data()
    l2i, i2l, labels_unique = _load_label_maps(train_df)
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("\u2713 Loaded embedding model: %s", EMBEDDING_MODEL_NAME)

    label_words = None
    try:
        label_words = build_label_words(train_df, labels_unique)
    except Exception as exc:
        logger.warning("TF-IDF word extraction failed: %s", exc)

    X_train, train_centroid, label_centroids = (None, None, {})
    if compute_centroids:
        X_train, train_centroid, label_centroids = _compute_centroids(
            emb_model, train_df, labels_unique)

    return LoadedAssets(
        clf=clf, emb_model=emb_model, train_df=train_df,
        label_to_idx=l2i, idx_to_label=i2l, labels_unique=labels_unique,
        label_words=label_words, X_train=X_train,
        train_centroid=train_centroid, label_centroids=label_centroids,
    )


# =============================================================================
# PREDICTION ON TARGET FILE
# =============================================================================

def _load_target_file():
    """Load and clean target file."""
    logger.info("Loading target file: %s", TARGET_FILE)
    if not os.path.exists(TARGET_FILE):
        raise FileNotFoundError(f"Target file not found: {TARGET_FILE}")
    df = pd.read_excel(TARGET_FILE)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Target file must have '{TEXT_COL}'. Found: {list(df.columns)}")
    df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("")
    logger.info("\u2713 Loaded target file: %d docs", len(df))
    return df


def _add_city_column(df):
    """Map source DataFrame identifiers to city names."""
    src_col = next((c for c in ["source_df", "_source_df"] if c in df.columns), None)
    df["City"] = df[src_col].map(CITY_MAPPING).fillna("Unknown") if src_col else "Unknown"
    return df


def _enrich_with_predictions(df, probs, idx_to_label):
    """Add prediction columns (label, confidence, second-best, margin, entropy)."""
    pred_idx = np.argmax(probs, axis=1).astype(int)
    pred_conf = probs[np.arange(len(probs)), pred_idx]
    second_idx = np.argsort(-probs, axis=1)[:, 1].astype(int)
    second_conf = probs[np.arange(len(probs)), second_idx]

    df["Predicted_label"] = [idx_to_label[int(i)] for i in pred_idx]
    df["Predicted_confidence"] = pred_conf
    df["Second_best_label"] = [idx_to_label[int(i)] for i in second_idx]
    df["Second_best_confidence"] = second_conf
    df["Prediction_margin"] = pred_conf - second_conf
    df["Prediction_entropy"] = prediction_entropy(probs)
    return df


def predict_target_file(assets):
    """Load target file, predict labels, return enriched DataFrame + embeddings + probs."""
    target_df = _load_target_file()
    X = encode_texts(assets.emb_model, target_df[TEXT_COL].tolist())
    probs = get_probabilities(assets.clf, X)
    target_df = _enrich_with_predictions(target_df, probs, assets.idx_to_label)
    target_df = _add_city_column(target_df)

    if STORE_FULL_PROBS_IN_DF:
        for j in range(probs.shape[1]):
            target_df[f"prob_{assets.idx_to_label[j]}"] = probs[:, j]

    logger.info("\u2713 Predictions complete")
    return target_df, X, probs


# =============================================================================
# DIAGNOSTICS: SUMMARIES
# =============================================================================

def summarize_predictions(target_df):
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 80)
    conf = target_df["Predicted_confidence"]
    summary = {
        "total_documents": len(target_df),
        "unique_labels": int(target_df["Predicted_label"].nunique()),
        "unique_cities": int(target_df["City"].nunique()),
        "avg_confidence": float(conf.mean()),
        "median_confidence": float(conf.median()),
        "min_confidence": float(conf.min()),
        "max_confidence": float(conf.max()),
        "std_confidence": float(conf.std()),
        "avg_margin": float(target_df["Prediction_margin"].mean()),
        "median_margin": float(target_df["Prediction_margin"].median()),
        "avg_entropy": float(target_df["Prediction_entropy"].mean()),
        "median_entropy": float(target_df["Prediction_entropy"].median()),
    }
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)

    label_counts = target_df["Predicted_label"].value_counts()
    logger.info("\nLabel distribution:")
    for label, count in label_counts.items():
        logger.info("  %s: %d (%.1f%%)", label, count, 100 * count / len(target_df))
    return summary, label_counts


def create_city_label_breakdown(target_df):
    breakdown = pd.crosstab(target_df["City"], target_df["Predicted_label"],
                            margins=True, margins_name="Total")
    breakdown_pct = pd.crosstab(target_df["City"], target_df["Predicted_label"],
                                normalize="index") * 100
    return breakdown, breakdown_pct


# =============================================================================
# DIAGNOSTICS: CONFIDENCE ANALYSIS
# =============================================================================

def analyze_confidence_thresholds(target_df, thresholds=DEFAULT_THRESHOLDS):
    n = len(target_df)
    results = []
    per_label_rows = []
    labels = sorted(target_df["Predicted_label"].unique().tolist())
    for t in thresholds:
        n_above = int((target_df["Predicted_confidence"] >= t).sum())
        pct_above = 100 * n_above / n if n else 0.0
        results.append({"Threshold": t, "Documents_Above": n_above,
                        "Percentage_Above": pct_above, "Documents_Below": n - n_above,
                        "Percentage_Below": 100 - pct_above})
        for lab in labels:
            df_lab = target_df[target_df["Predicted_label"] == lab]
            m = len(df_lab)
            n_lab_above = int((df_lab["Predicted_confidence"] >= t).sum()) if m else 0
            per_label_rows.append({"Label": lab, "Threshold": t, "N_Label": m,
                                   "N_Above": n_lab_above,
                                   "Pct_Above": 100 * n_lab_above / m if m else np.nan})
    return pd.DataFrame(results), pd.DataFrame(per_label_rows)


def analyze_confidence_by_city(target_df):
    rows = []
    for city in sorted(target_df["City"].unique()):
        df = target_df[target_df["City"] == city]
        conf = df["Predicted_confidence"]
        rows.append({
            "City": city, "N_Documents": len(df),
            "Avg_Confidence": float(conf.mean()), "Median_Confidence": float(conf.median()),
            "Std_Confidence": float(conf.std()), "Min_Confidence": float(conf.min()),
            "Max_Confidence": float(conf.max()),
            "Q25_Confidence": float(conf.quantile(0.25)),
            "Q75_Confidence": float(conf.quantile(0.75)),
            "Avg_Margin": float(df["Prediction_margin"].mean()),
            "Avg_Entropy": float(df["Prediction_entropy"].mean()),
        })
    return pd.DataFrame(rows)


# =============================================================================
# VALIDATION: THRESHOLD SWEEP (per city)
# =============================================================================

def threshold_sweep(df_pred, thresholds=DEFAULT_THRESHOLDS):
    rows_city = []
    rows_city_label = []
    for thr in thresholds:
        below = df_pred["Predicted_confidence"] < thr
        for city, grp in df_pred.groupby("City", dropna=False):
            mask = below[grp.index]
            n = len(grp)
            rows_city.append({"threshold": thr, "city": city, "n": n,
                              "n_below": int(mask.sum()),
                              "pct_below": (int(mask.sum()) / n) if n else 0.0,
                              "conf_median": float(grp["Predicted_confidence"].median()),
                              "conf_mean": float(grp["Predicted_confidence"].mean())})
        for (city, lab), grp in df_pred.groupby(["City", "Predicted_label"], dropna=False):
            mask = below[grp.index]
            n = len(grp)
            rows_city_label.append({"threshold": thr, "city": city, "pred_label": lab,
                                    "n": n, "n_below": int(mask.sum()),
                                    "pct_below": float(mask.mean()) if n else 0.0})
    return pd.DataFrame(rows_city), pd.DataFrame(rows_city_label)


# =============================================================================
# VALIDATION: CITY SIMILARITY / OOD DIAGNOSTICS
# =============================================================================

def _city_ood_stats(grp, idx, X_target, assets):
    """Compute OOD stats for one city."""
    X_city = X_target[idx]
    city_centroid = X_city.mean(axis=0)
    dist_to_train = float(cosine_distances(
        city_centroid.reshape(1, -1), assets.train_centroid.reshape(1, -1))[0, 0])

    nearest_label, dist_to_nearest = None, np.nan
    lc = assets.label_centroids
    if lc:
        label_names = list(lc.keys())
        label_mat = np.vstack([lc[l] for l in label_names])
        dists = cosine_distances(city_centroid.reshape(1, -1), label_mat)[0]
        j = int(np.argmin(dists))
        nearest_label, dist_to_nearest = label_names[j], float(dists[j])

    doc_dists = [
        float(cosine_distances(X_target[i].reshape(1, -1),
              lc[grp.loc[i, "Predicted_label"]].reshape(1, -1))[0, 0])
        for i in idx if grp.loc[i, "Predicted_label"] in lc
    ]
    return {
        "dist_to_train_centroid": dist_to_train,
        "nearest_label_centroid": nearest_label,
        "dist_to_nearest_label_centroid": dist_to_nearest,
        "mean_doc_dist_to_pred_centroid": float(np.mean(doc_dists)) if doc_dists else np.nan,
        "median_doc_dist_to_pred_centroid": float(np.median(doc_dists)) if doc_dists else np.nan,
    }


def compute_city_similarity(df_pred, X_target, assets):
    """Per-city OOD diagnostics: distance to training centroid + per-doc distances."""
    rows = []
    for city, grp in df_pred.groupby("City", dropna=False):
        idx = grp.index.values
        stats = _city_ood_stats(grp, idx, X_target, assets)
        rows.append({"city": city, "n_docs": len(grp), **stats,
                     "confidence_mean": float(grp["Predicted_confidence"].mean()),
                     "confidence_median": float(grp["Predicted_confidence"].median())})
    return pd.DataFrame(rows).sort_values("dist_to_train_centroid").reset_index(drop=True)


# =============================================================================
# VALIDATION: HOLDOUT EVALUATION
# =============================================================================

def _load_holdout_file(path, holdout_name, assets):
    """Load holdout file, normalise labels, drop unknowns. Returns (df, y_true) or None."""
    if not os.path.exists(path):
        logger.info("%s not found (%s). Skipping.", holdout_name, path)
        return None

    df = pd.read_excel(path)
    if DOC_COL not in df.columns or LABEL_COL not in df.columns:
        logger.warning("%s: missing expected columns. Skipping.", holdout_name)
        return None

    df[DOC_COL] = df[DOC_COL].astype(str).fillna("")
    df[LABEL_COL] = df[LABEL_COL].astype(str).map(normalize_label)
    df = df[df[DOC_COL].str.len() > 0].reset_index(drop=True)

    y_mapped = df[LABEL_COL].map(assets.label_to_idx)
    keep = y_mapped.notna()
    if keep.sum() == 0:
        logger.warning("%s: no labels overlap with classifier. Skipping.", holdout_name)
        return None
    if (~keep).sum() > 0:
        logger.warning("%s: dropping %d docs with unknown labels.", holdout_name, (~keep).sum())
    df = df[keep].reset_index(drop=True)
    return df, y_mapped[keep].astype(int).values


def _calibration_bins(y_true, y_pred, confidence, n_bins=CALIBRATION_N_BINS):
    """Compute binned calibration data for reliability diagram."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    correct = (y_true == y_pred).astype(float)
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if hi == 1.0:
            mask = mask | (confidence == 1.0)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({"bin_lo": lo, "bin_hi": hi, "bin_mid": (lo + hi) / 2,
                     "mean_confidence": float(confidence[mask].mean()),
                     "accuracy": float(correct[mask].mean()), "n": n})
    return pd.DataFrame(rows)


def evaluate_holdout(path, assets, holdout_name):
    """Evaluate classifier on a holdout file. Returns metrics dict or None."""
    loaded = _load_holdout_file(path, holdout_name, assets)
    if loaded is None:
        return None
    df, y_true = loaded

    X = encode_texts(assets.emb_model, df[DOC_COL].tolist())
    probs = assets.clf.predict_proba(X)
    y_pred = np.argmax(probs, axis=1).astype(int)
    conf = probs[np.arange(len(probs)), y_pred]

    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    logger.info("%s | n=%d | F1w=%.4f | Acc=%.4f", holdout_name, len(y_true), f1w, acc)

    present = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    present_names = [assets.idx_to_label[i] for i in present]

    return {
        "holdout": holdout_name, "n": int(len(y_true)),
        "f1_weighted": float(f1w), "accuracy": float(acc),
        "report": classification_report(y_true, y_pred, labels=present,
                                        target_names=present_names, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=present),
        "labels_used": present, "label_names_used": present_names,
        "calibration": _calibration_bins(y_true, y_pred, conf),
    }


# =============================================================================
# CONFUSION ANALYSIS
# =============================================================================

def create_confusion_matrix(target_df, labels_unique):
    """Confusion matrix: top-1 prediction vs top-2 prediction."""
    l2i = {lab: i for i, lab in enumerate(labels_unique)}
    n = len(labels_unique)
    mat = np.zeros((n, n), dtype=int)
    for first, second in zip(target_df["Predicted_label"], target_df["Second_best_label"]):
        if first in l2i and second in l2i:
            mat[l2i[first], l2i[second]] += 1
    return pd.DataFrame(mat, index=labels_unique, columns=labels_unique)


def top_confusions(df_confusion, top_k=25):
    rows = []
    labels = df_confusion.index.tolist()
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j:
                c = int(df_confusion.iloc[i, j])
                if c > 0:
                    rows.append({"First": a, "Second": b, "Count": c})
    return pd.DataFrame(rows).sort_values("Count", ascending=False).head(top_k).reset_index(drop=True)


# =============================================================================
# WORD EXTRACTION HELPERS
# =============================================================================

def extract_top_words_per_label(label_words, labels_unique, top_n=20):
    rows = []
    for lab in labels_unique:
        for rank, (word, score) in enumerate(label_words.get(lab, [])[:top_n], start=1):
            rows.append({"Label": lab, "Rank": rank, "Word": word, "Score": float(score)})
    return pd.DataFrame(rows)


def _build_tfidf_score_matrix(label_words, labels_unique, candidate_k):
    """Build word×label TF-IDF score matrix and vocabulary list."""
    per_label = {}
    vocab = set()
    for lab in labels_unique:
        words = label_words.get(lab, [])[:candidate_k]
        per_label[lab] = dict(words)
        vocab.update(w for w, _ in words)
    vocab_list = sorted(vocab)
    M = np.zeros((len(labels_unique), len(vocab_list)))
    for r, lab in enumerate(labels_unique):
        for c, w in enumerate(vocab_list):
            if w in per_label.get(lab, {}):
                M[r, c] = per_label[lab][w]
    return M, vocab_list


def extract_discriminative_words(label_words, labels_unique, top_n=15, candidate_k=60):
    """Contrastive word extraction: words with highest TF-IDF relative to other labels."""
    if not any(label_words.get(lab) for lab in labels_unique):
        return pd.DataFrame()

    M, vocab_list = _build_tfidf_score_matrix(label_words, labels_unique, candidate_k)
    rows = []
    for r, lab in enumerate(labels_unique):
        scores = M[r]
        others_mean = (M.sum(axis=0) - scores) / max(1, M.shape[0] - 1)
        disc = scores - others_mean
        for j in np.argsort(-disc)[:top_n]:
            if disc[j] <= 0:
                continue
            rows.append({"Label": lab, "Word": vocab_list[j],
                         "DiscriminationScore": float(disc[j]),
                         "Score_Label": float(scores[j]),
                         "Score_OthersMean": float(others_mean[j])})
    return pd.DataFrame(rows).sort_values(["Label", "DiscriminationScore"], ascending=[True, False])


def top_words_wide_table(df_top_words, top_n=20):
    if df_top_words.empty:
        return pd.DataFrame()
    df = df_top_words.sort_values(["Label", "Rank"]).copy()
    wide = (df[df["Rank"] <= top_n]
            .pivot(index="Label", columns="Rank", values="Word")
            .rename(columns=lambda r: f"Top{int(r)}").reset_index())
    csv = (df[df["Rank"] <= top_n]
           .groupby("Label")["Word"].apply(lambda x: ", ".join(x.tolist()))
           .reset_index(name="TopWordsCSV"))
    return wide.merge(csv, on="Label", how="left")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def _safe_plot(name, fn):
    try:
        fn()
    except Exception as exc:
        logger.warning("Plot '%s' failed: %s", name, exc)


def plot_confidence_threshold_analysis(df_thr, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df_thr["Threshold"], df_thr["Percentage_Above"], marker="o", linewidth=2)
    axes[0].fill_between(df_thr["Threshold"], df_thr["Percentage_Above"], alpha=0.3)
    axes[0].set(xlabel="Confidence Threshold", ylabel="Percentage (%)",
                title="Documents Above Confidence Threshold", ylim=[0, 105])
    axes[0].grid(True, alpha=0.3)
    key = df_thr[df_thr["Threshold"].isin([0.5, 0.6, 0.7, 0.8, 0.9])].copy()
    x = np.arange(len(key))
    w = 0.35
    axes[1].bar(x - w / 2, key["Documents_Above"], w, label="Above", alpha=0.7)
    axes[1].bar(x + w / 2, key["Documents_Below"], w, label="Below", alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{t:.2f}" for t in key["Threshold"]])
    axes[1].set(xlabel="Threshold", ylabel="Count", title="Above/Below Key Thresholds")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_threshold_analysis.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: confidence_threshold_analysis.png")


def plot_confidence_boxplot_by_city(target_df, output_dir, exclude_other=True):
    df = _filter_excluded(target_df) if exclude_other else target_df
    if df.empty:
        return
    suffix = "_excl_other" if exclude_other else ""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cities = sorted(df["City"].unique())
    data = [df[df["City"] == c]["Predicted_confidence"].values for c in cities]
    axes[0].boxplot(data, labels=cities)
    axes[0].set(xlabel="City", ylabel="Confidence", title="Confidence by City (Boxplot)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].violinplot(data, positions=range(len(cities)), showmeans=True, showmedians=True)
    axes[1].set_xticks(range(len(cities)))
    axes[1].set_xticklabels(cities, rotation=45, ha="right")
    axes[1].set(xlabel="City", ylabel="Confidence", title="Confidence by City (Violin)")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confidence_by_city{suffix}.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: confidence_by_city%s.png", suffix)


def plot_confidence_boxplot_by_label(target_df, output_dir, min_n=20, exclude_other=True):
    df = _filter_excluded(target_df) if exclude_other else target_df
    if df.empty:
        return
    suffix = "_excl_other" if exclude_other else ""
    labels_sorted = (df.groupby("Predicted_label")["Predicted_confidence"]
                     .median().sort_values().index.tolist())
    labels_sorted = [l for l in labels_sorted if (df["Predicted_label"] == l).sum() >= min_n]
    data = [df[df["Predicted_label"] == l]["Predicted_confidence"].values for l in labels_sorted]
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels_sorted, vert=True)
    plt.xlabel("Predicted Label"); plt.ylabel("Confidence")
    plt.title("Confidence Distribution by Label")
    plt.xticks(rotation=45, ha="right"); plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confidence_by_label{suffix}.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: confidence_by_label%s.png", suffix)


def plot_confidence_distribution(target_df, output_dir, exclude_other=True):
    df = _filter_excluded(target_df) if exclude_other else target_df
    if df.empty:
        return
    suffix = "_excl_other" if exclude_other else ""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    conf = df["Predicted_confidence"]
    axes[0].hist(conf, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(conf.median(), linestyle="--", linewidth=2, label=f"Median: {conf.median():.3f}")
    axes[0].axvline(conf.mean(), linestyle=":", linewidth=2, label=f"Mean: {conf.mean():.3f}")
    axes[0].set(xlabel="Confidence", ylabel="Count", title="Overall Confidence Distribution")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    labels_sorted = df.groupby("Predicted_label")["Predicted_confidence"].median().sort_values().index
    data = [df[df["Predicted_label"] == l]["Predicted_confidence"].values for l in labels_sorted]
    axes[1].boxplot(data, labels=labels_sorted, vert=True)
    axes[1].set(xlabel="Label", ylabel="Confidence", title="Confidence by Label")
    axes[1].tick_params(axis="x", rotation=45); axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confidence_distribution{suffix}.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: confidence_distribution%s.png", suffix)


def plot_label_distribution(label_counts, output_dir, exclude_other=True):
    lc = label_counts.copy()
    if exclude_other:
        lc = lc.drop(labels=list(EXCLUDE_LABELS_IN_VIZ), errors="ignore")
    if lc.empty:
        return
    suffix = "_excl_other" if exclude_other else ""
    plt.figure(figsize=(12, 6))
    lc.sort_values(ascending=True).plot(kind="barh")
    plt.xlabel("Number of Documents"); plt.ylabel("Label")
    plt.title("Predicted Label Distribution"); plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"label_distribution{suffix}.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: label_distribution%s.png", suffix)


def plot_city_label_heatmap(breakdown, output_dir, exclude_other=True):
    if not _HAS_SEABORN:
        return
    bc = breakdown.drop("Total", axis=0, errors="ignore").drop("Total", axis=1, errors="ignore")
    if exclude_other:
        bc = bc.drop(columns=list(EXCLUDE_LABELS_IN_VIZ), errors="ignore")
    if bc.empty:
        return
    suffix = "_excl_other" if exclude_other else ""
    plt.figure(figsize=(14, 8))
    sns.heatmap(bc, annot=True, fmt="d", cmap=CMAP_BLUE_YELLOW, linewidths=0.5)
    plt.xlabel("Predicted Label"); plt.ylabel("City"); plt.title("City \u00d7 Label Distribution")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"city_label_heatmap{suffix}.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: city_label_heatmap%s.png", suffix)


def plot_confidence_by_city_and_label(target_df, output_dir, exclude_other=True):
    df = _filter_excluded(target_df) if exclude_other else target_df
    pivot = df.pivot_table(values="Predicted_confidence", index="City", columns="Predicted_label", aggfunc="mean")
    if exclude_other:
        pivot = pivot.drop(columns=list(EXCLUDE_LABELS_IN_VIZ), errors="ignore")
    suffix = "_excl_other" if exclude_other else ""
    if _HAS_SEABORN and not pivot.empty:
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=CMAP_TURQUOISE,
                    center=pivot.stack().median())
        plt.xlabel("Predicted Label"); plt.ylabel("City")
        plt.title("Avg Confidence by City and Label")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confidence_by_city_and_label{suffix}.png"), dpi=DPI, bbox_inches="tight")
        plt.close()
        logger.info("\u2713 Saved: confidence_by_city_and_label%s.png", suffix)
    return pivot


def plot_confusion_matrix(df_confusion, output_dir, exclude_other=True):
    if not _HAS_SEABORN:
        return
    dfc = df_confusion.copy()
    if exclude_other:
        dfc = dfc.drop(index=list(EXCLUDE_LABELS_IN_VIZ), errors="ignore")
        dfc = dfc.drop(columns=list(EXCLUDE_LABELS_IN_VIZ), errors="ignore")
    if dfc.empty:
        return
    suffix = "_excl_other" if exclude_other else ""
    df_norm = dfc.div(dfc.sum(axis=1).replace(0, np.nan), axis=0) * 100
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_norm, annot=True, fmt=".1f", cmap=CMAP_BLUE_YELLOW, linewidths=0.5)
    plt.xlabel("Second-Best Prediction"); plt.ylabel("First Prediction")
    plt.title("Label Confusion Matrix (% of first predictions with each second-best)")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"label_confusion_matrix{suffix}.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: label_confusion_matrix%s.png", suffix)


def plot_confusion_network(target_df, labels_unique, output_dir, min_count=CONFUSION_NETWORK_MIN_COUNT):
    if not _HAS_NETWORKX:
        return
    pairs = [(a, b) for a, b in zip(target_df["Predicted_label"], target_df["Second_best_label"]) if a != b]
    counts = Counter(pairs)
    edges = [(a, b, c) for (a, b), c in counts.items() if c >= min_count]
    if not edges:
        return
    G = nx.DiGraph()
    for lab in labels_unique:
        G.add_node(lab)
    for a, b, c in edges:
        G.add_edge(a, b, weight=c)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    plt.figure(figsize=(14, 12))
    nx.draw_networkx_nodes(G, pos, node_size=2800, alpha=0.9, edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    maxw = max(weights) if weights else 1
    for u, v in G.edges():
        w = G[u][v]["weight"]
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=5 * (w / maxw),
                               alpha=0.6, arrowsize=18, arrowstyle="->")
    edge_labels = {(u, v): str(G[u][v]["weight"]) for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(f"Label Confusion Network (\u2265{min_count} docs)", fontsize=14, fontweight="bold")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_network.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: confusion_network.png")


def plot_label_centroid_distances(train_df, emb_model, labels_unique, output_dir, exclude_other=True):
    labels_viz = _labels_without_excluded(labels_unique) if exclude_other else labels_unique
    if not labels_viz:
        return None
    suffix = "_excl_other" if exclude_other else ""
    X = encode_texts(emb_model, train_df[DOC_COL].astype(str).fillna("").tolist())
    y = train_df[LABEL_COL].values
    centroids = []
    for lab in labels_viz:
        mask = y == lab
        centroids.append(X[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(X.shape[1]))
    centroids = np.vstack(centroids)
    D = cosine_distances(centroids)
    if _HAS_SEABORN:
        plt.figure(figsize=(12, 10))
        sns.heatmap(D, annot=True, fmt=".3f", cmap=CMAP_TURQUOISE_REV,
                    xticklabels=labels_viz, yticklabels=labels_viz, vmin=0, vmax=1)
        plt.title("Cosine Distance Between Label Centroids"); plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"label_centroid_distances{suffix}.png"), dpi=DPI, bbox_inches="tight")
        plt.close()
        logger.info("\u2713 Saved: label_centroid_distances%s.png", suffix)
    return D


def plot_embeddings_2d(train_df, emb_model, labels_unique, output_dir,
                       n_per_class=EMBED_VIZ_N_PER_CLASS, reducer=EMBED_VIZ_REDUCER):
    logger.info("Creating 2D embedding visualization (reducer=%s)\u2026", reducer)
    sampled = (train_df.groupby(LABEL_COL, group_keys=False)
               .apply(lambda g: g.sample(min(n_per_class, len(g)), random_state=42))
               .reset_index(drop=True))
    X = encode_texts(emb_model, sampled[DOC_COL].astype(str).fillna("").tolist())
    reducer = reducer.lower().strip()
    if reducer == "umap" and _HAS_UMAP:
        X2 = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)
        red_name = "umap"
    else:
        X2 = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(X)
        red_name = "tsne"
    plt.figure(figsize=(16, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels_unique)))
    for i, lab in enumerate(labels_unique):
        mask = sampled[LABEL_COL].values == lab
        plt.scatter(X2[mask, 0], X2[mask, 1], s=25, alpha=0.6, label=lab,
                    color=colors[i], edgecolors="black", linewidth=0.2)
    plt.xlabel("Component 1"); plt.ylabel("Component 2")
    plt.title(f"2D Embedding Visualization ({red_name.upper()})", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2, frameon=True)
    plt.grid(True, alpha=0.2); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"embeddings_2d_{red_name}.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: embeddings_2d_%s.png", red_name)


def plot_top_words_per_label(df_top_words, labels_unique, output_dir, top_n=15, exclude_other=True):
    if df_top_words is None or df_top_words.empty:
        return
    labels_viz = _labels_without_excluded(labels_unique) if exclude_other else labels_unique
    if not labels_viz:
        return
    suffix = "_excl_other" if exclude_other else ""
    n_cols = 3
    n_rows = int(np.ceil(len(labels_viz) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = np.array(axes).flatten()
    for idx, lab in enumerate(labels_viz):
        ax = axes[idx]
        ld = df_top_words[df_top_words["Label"] == lab].sort_values("Rank").head(top_n)
        if ld.empty:
            ax.text(0.5, 0.5, f"No data for\n{lab}", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.barh(ld["Word"].tolist()[::-1], ld["Score"].tolist()[::-1], alpha=0.85, edgecolor="black")
        ax.set_xlabel("TF-IDF Score", fontsize=9)
        ax.set_title(lab, fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8); ax.grid(True, alpha=0.3, axis="x")
    for j in range(len(labels_viz), len(axes)):
        axes[j].axis("off")
    plt.suptitle("Top Words per Label (TF-IDF)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"top_words_per_label{suffix}.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: top_words_per_label%s.png", suffix)


def plot_discriminative_words_per_label(df_disc, labels_unique, output_dir, top_n=15, exclude_other=True):
    if df_disc is None or df_disc.empty:
        return
    labels_viz = _labels_without_excluded(labels_unique) if exclude_other else labels_unique
    if not labels_viz:
        return
    suffix = "_excl_other" if exclude_other else ""
    n_cols = 3
    n_rows = int(np.ceil(len(labels_viz) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = np.array(axes).flatten()
    for idx, lab in enumerate(labels_viz):
        ax = axes[idx]
        ld = df_disc[df_disc["Label"] == lab].head(top_n)
        if ld.empty:
            ax.text(0.5, 0.5, f"No data for\n{lab}", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.barh(ld["Word"].tolist()[::-1], ld["DiscriminationScore"].tolist()[::-1], alpha=0.8, edgecolor="black")
        ax.set_xlabel("Discrimination Score", fontsize=9)
        ax.set_title(lab, fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8); ax.grid(True, alpha=0.3, axis="x")
    for j in range(len(labels_viz), len(axes)):
        axes[j].axis("off")
    plt.suptitle("Discriminative Words per Label (contrastive c-TF-IDF)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"discriminative_words_per_label{suffix}.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: discriminative_words_per_label%s.png", suffix)


def plot_threshold_by_city(df_city_thr, output_dir, ref_threshold=0.60):
    if df_city_thr.empty:
        return
    avail = df_city_thr["threshold"].unique()
    ref = ref_threshold if ref_threshold in avail else float(avail[len(avail) // 2])
    sub = df_city_thr[df_city_thr["threshold"] == ref].sort_values("pct_below", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.barh(sub["city"].astype(str), sub["pct_below"].values, alpha=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel(f"Fraction with confidence < {ref:.2f}")
    plt.title(f"Low-confidence share by city (threshold={ref:.2f})")
    plt.grid(axis="x", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pct_below_{ref:.2f}_by_city.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: pct_below_%.2f_by_city.png", ref)


def plot_city_similarity(df_sim, output_dir):
    if df_sim.empty:
        return
    sub = df_sim.sort_values("dist_to_train_centroid", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.barh(sub["city"].astype(str), sub["dist_to_train_centroid"].values, alpha=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("Cosine distance to training centroid")
    plt.title("City embedding distance to training distribution")
    plt.grid(axis="x", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "city_distance_to_training_centroid.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: city_distance_to_training_centroid.png")


def plot_calibration(cal_df, holdout_name, output_dir):
    if cal_df.empty:
        return
    mid = cal_df["mean_confidence"].values
    acc = cal_df["accuracy"].values
    n = cal_df["n"].values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.bar(mid, acc, width=1.0 / CALIBRATION_N_BINS * 0.8, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Actual accuracy"); ax1.set_title(f"Calibration: {holdout_name}")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.legend(); ax1.grid(alpha=0.25)
    ax2.bar(mid, n, width=1.0 / CALIBRATION_N_BINS * 0.8, alpha=0.5, color="gray", edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Mean predicted confidence"); ax2.set_ylabel("Count"); ax2.grid(alpha=0.25)
    plt.tight_layout()
    safe_name = holdout_name.replace(" ", "_").lower()
    plt.savefig(os.path.join(output_dir, f"calibration_{safe_name}.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: calibration_%s.png", safe_name)


def plot_ood_distances(df_pred, X_target, assets, output_dir, n_sample=5000):
    label_centroids = assets.label_centroids
    dists = np.full(len(df_pred), np.nan)
    for i in range(len(df_pred)):
        lab = df_pred.iloc[i]["Predicted_label"]
        if lab in label_centroids:
            dists[i] = float(cosine_distances(
                X_target[i].reshape(1, -1), label_centroids[lab].reshape(1, -1))[0, 0])
    valid = ~np.isnan(dists)
    if valid.sum() == 0:
        return
    conf = df_pred["Predicted_confidence"].values[valid]
    d = dists[valid]
    if len(d) > n_sample:
        idx = np.random.default_rng(42).choice(len(d), n_sample, replace=False)
        d, conf = d[idx], conf[idx]
    plt.figure(figsize=(10, 6))
    plt.scatter(conf, d, alpha=0.15, s=8, c="steelblue")
    plt.xlabel("Predicted confidence"); plt.ylabel("Cosine distance to predicted label centroid")
    plt.title("OOD signal: distance to label centroid vs confidence")
    plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ood_distance_vs_confidence.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("\u2713 Saved: ood_distance_vs_confidence.png")


# =============================================================================
# MAIN
# =============================================================================

def _save_excel(df, name, index=False):
    """Save DataFrame to OUTPUTS_DIR."""
    path = os.path.join(OUTPUTS_DIR, name)
    df.to_excel(path, index=index)
    logger.info("\u2713 Saved: %s", name)


def _run_inference(assets):
    """Predict on target file and save."""
    target_df, X_target, probs = predict_target_file(assets)
    _save_excel(target_df, "full_predictions.xlsx")
    return target_df, X_target, probs


def _run_summary_analyses(target_df):
    """Run summary statistics, breakdown, and confidence analyses."""
    summary, label_counts = summarize_predictions(target_df)

    breakdown, breakdown_pct = create_city_label_breakdown(target_df)
    with pd.ExcelWriter(os.path.join(OUTPUTS_DIR, "city_label_breakdown.xlsx")) as w:
        breakdown.to_excel(w, sheet_name="Counts")
        breakdown_pct.to_excel(w, sheet_name="Percentages")
    logger.info("\u2713 Saved: city_label_breakdown.xlsx")

    df_thr, df_thr_label = analyze_confidence_thresholds(target_df)
    with pd.ExcelWriter(os.path.join(OUTPUTS_DIR, "confidence_thresholds_analysis.xlsx")) as w:
        df_thr.to_excel(w, sheet_name="Overall", index=False)
        if df_thr_label is not None and not df_thr_label.empty:
            df_thr_label.to_excel(w, sheet_name="PerLabel", index=False)
    logger.info("\u2713 Saved: confidence_thresholds_analysis.xlsx")

    _save_excel(analyze_confidence_by_city(target_df), "confidence_by_city.xlsx")
    return summary, label_counts, breakdown, df_thr


def _run_threshold_sweep(target_df):
    """Per-city and per-city×label threshold sweep."""
    df_city_thr, df_city_label_thr = threshold_sweep(target_df)
    _save_excel(df_city_thr, "threshold_sweep_by_city.xlsx")
    _save_excel(df_city_label_thr, "threshold_sweep_by_city_and_label.xlsx")
    return df_city_thr


def _run_city_similarity(target_df, X_target, assets):
    """City-level OOD diagnostics."""
    if assets.train_centroid is None:
        return pd.DataFrame()
    df_sim = compute_city_similarity(target_df, X_target, assets)
    _save_excel(df_sim, "city_similarity_to_training.xlsx")
    return df_sim


def _run_confusion_analysis(target_df, labels_unique):
    """Top-1 vs top-2 confusion matrix."""
    df_confusion = create_confusion_matrix(target_df, labels_unique)
    df_confusion.to_excel(os.path.join(OUTPUTS_DIR, "label_confusion_matrix.xlsx"))
    _save_excel(top_confusions(df_confusion, top_k=50), "top_label_confusions.xlsx")
    return df_confusion


def _run_word_extraction(assets):
    """TF-IDF word extraction (top words + discriminative)."""
    if assets.label_words is None:
        logger.warning("No label words \u2014 skipping word extraction.")
        return None, None
    try:
        df_top = extract_top_words_per_label(assets.label_words, assets.labels_unique)
        _save_excel(df_top, "top_words_per_label_long.xlsx")
        _save_excel(top_words_wide_table(df_top), "top_words_per_label_wide.xlsx")
        df_disc = extract_discriminative_words(assets.label_words, assets.labels_unique)
        _save_excel(df_disc, "discriminative_words.xlsx")
        return df_top, df_disc
    except Exception as exc:
        logger.warning("Word extraction failed: %s", exc)
        return None, None


def _run_holdout_evaluation(assets):
    """Evaluate on static + dynamic holdouts; save reports and calibration."""
    holdout_rows = []
    for path, name in [(STATIC_HOLDOUT_PATH, "Static Holdout"),
                       (DYNAMIC_HOLDOUT_PATH, "Dynamic Holdout")]:
        res = evaluate_holdout(path, assets, name)
        if res is None:
            continue
        holdout_rows.append({"holdout": res["holdout"], "n": res["n"],
                             "f1_weighted": res["f1_weighted"], "accuracy": res["accuracy"]})
        safe = name.replace(" ", "_").lower()
        pd.DataFrame(res["confusion_matrix"], columns=res["label_names_used"],
                     index=res["label_names_used"]).to_excel(
            os.path.join(OUTPUTS_DIR, f"confusion_matrix_{safe}.xlsx"))
        with open(os.path.join(OUTPUTS_DIR, f"classification_report_{safe}.txt"), "w") as f:
            f.write(res["report"])
        cal = res["calibration"]
        if not cal.empty:
            _save_excel(cal, f"calibration_data_{safe}.xlsx")
            _safe_plot(f"calibration {name}", lambda: plot_calibration(cal, name, OUTPUTS_DIR))

    if holdout_rows:
        _save_excel(pd.DataFrame(holdout_rows), "holdout_evaluation_summary.xlsx")


def _build_diagnostics_summary(target_df):
    """Build and save the diagnostics summary table."""
    conf = target_df["Predicted_confidence"]
    df_summary = pd.DataFrame({
        "Metric": ["Total Documents", "Unique Labels", "Unique Cities",
                    "Avg Confidence", "Median Confidence", "Std Confidence",
                    "Min Confidence", "Max Confidence",
                    "Avg Margin (top1\u2212top2)", "Median Margin",
                    "Avg Entropy", "Median Entropy",
                    "Docs \u2265 0.6", "Docs \u2265 0.7", "Docs \u2265 0.8"],
        "Value": [len(target_df), target_df["Predicted_label"].nunique(),
                  target_df["City"].nunique(),
                  f"{conf.mean():.4f}", f"{conf.median():.4f}", f"{conf.std():.4f}",
                  f"{conf.min():.4f}", f"{conf.max():.4f}",
                  f"{target_df['Prediction_margin'].mean():.4f}",
                  f"{target_df['Prediction_margin'].median():.4f}",
                  f"{target_df['Prediction_entropy'].mean():.4f}",
                  f"{target_df['Prediction_entropy'].median():.4f}",
                  f"{(conf >= 0.6).sum()} ({100 * (conf >= 0.6).mean():.1f}%)",
                  f"{(conf >= 0.7).sum()} ({100 * (conf >= 0.7).mean():.1f}%)",
                  f"{(conf >= 0.8).sum()} ({100 * (conf >= 0.8).mean():.1f}%)"],
    })
    _save_excel(df_summary, "diagnostics_summary.xlsx")


def _run_all_plots(target_df, X_target, assets, label_counts, breakdown,
                   df_thr, df_city_thr, df_sim, df_confusion, df_top_words, df_disc_words):
    """Run all plotting functions with safe error handling."""
    lu = assets.labels_unique
    plots = [
        ("confidence threshold", lambda: plot_confidence_threshold_analysis(df_thr, OUTPUTS_DIR)),
        ("confidence by city", lambda: plot_confidence_boxplot_by_city(target_df, OUTPUTS_DIR)),
        ("confidence by label", lambda: plot_confidence_boxplot_by_label(target_df, OUTPUTS_DIR)),
        ("confidence distribution", lambda: plot_confidence_distribution(target_df, OUTPUTS_DIR)),
        ("label distribution", lambda: plot_label_distribution(label_counts, OUTPUTS_DIR)),
        ("city \u00d7 label heatmap", lambda: plot_city_label_heatmap(breakdown, OUTPUTS_DIR)),
        ("confidence by city & label", lambda: plot_confidence_by_city_and_label(target_df, OUTPUTS_DIR)),
        ("confusion matrix", lambda: plot_confusion_matrix(df_confusion, OUTPUTS_DIR)),
        ("confusion network", lambda: plot_confusion_network(target_df, lu, OUTPUTS_DIR)),
        ("label centroid distances", lambda: plot_label_centroid_distances(
            assets.train_df, assets.emb_model, lu, OUTPUTS_DIR)),
        ("2D embeddings", lambda: plot_embeddings_2d(assets.train_df, assets.emb_model, lu, OUTPUTS_DIR)),
        ("top words per label", lambda: plot_top_words_per_label(df_top_words, lu, OUTPUTS_DIR)),
        ("discriminative words", lambda: plot_discriminative_words_per_label(df_disc_words, lu, OUTPUTS_DIR)),
        ("threshold by city", lambda: plot_threshold_by_city(df_city_thr, OUTPUTS_DIR)),
        ("city similarity", lambda: plot_city_similarity(df_sim, OUTPUTS_DIR)),
        ("OOD distances", lambda: plot_ood_distances(target_df, X_target, assets, OUTPUTS_DIR)),
    ]
    for name, fn in plots:
        _safe_plot(name, fn)


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    logger.info("=" * 80)
    logger.info("CITY-MOVE | DIAGNOSTICS & VALIDATION v4.0")
    logger.info("=" * 80)

    assets = load_assets(compute_centroids=True)
    target_df, X_target, probs = _run_inference(assets)
    summary, label_counts, breakdown, df_thr = _run_summary_analyses(target_df)
    df_city_thr = _run_threshold_sweep(target_df)
    df_sim = _run_city_similarity(target_df, X_target, assets)
    df_confusion = _run_confusion_analysis(target_df, assets.labels_unique)
    df_top_words, df_disc_words = _run_word_extraction(assets)
    _run_holdout_evaluation(assets)
    _run_all_plots(target_df, X_target, assets, label_counts, breakdown,
                   df_thr, df_city_thr, df_sim, df_confusion, df_top_words, df_disc_words)
    _build_diagnostics_summary(target_df)

    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTICS & VALIDATION COMPLETE!")
    logger.info("Outputs in: %s", OUTPUTS_DIR)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
