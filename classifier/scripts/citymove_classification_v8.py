#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CITY-MOVE | Iterative Supervised Classification System
Version: 8.0

Changes from v6:
─────────────────
1. Reverted keyword feature experiment (v7) — empirically degraded
   confidence on previously well-classified docs.
2. Boilerplate stripping (extract_discriminative_text): exploratory
   preprocessing step that extracts title portion before first colon
   if ≥15 chars. May or may not be active in final production run.
3. Progressive confidence threshold support (0.50 → 0.60 → 0.70).
4. 8 classes: Total PA and Recreational PA kept as separate categories.

Author: Oberon — Mar 2026
"""

from __future__ import annotations

import os
import platform
import logging
import re
import shutil
from collections import namedtuple
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import joblib
from sentence_transformers import SentenceTransformer

from synthetic_text_generation import generate_synthetic_data


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

# ── Input files ──────────────────────────────────────────────────────────────
TARGET_FILE = os.path.join(DATA_DIR, "to_predict_df.xlsx")
BASE_TRAIN_FILE = os.path.join(DATA_DIR, "sampled_docs_df_04022026.xlsx")

BASE_DOCS_COL = "docs"
BASE_LABELS_COL = "labels"

# ── Standard column names ────────────────────────────────────────────────────
DOC_COL = "docs"
LABEL_COL = "labels"
TEXT_COL = "doc"
IS_SYNTHETIC_COL = "is_synthetic"
DOC_GROUP_COL = "doc_group_id"

# ── Active learning ──────────────────────────────────────────────────────────
# NOTE: Confidence threshold was raised progressively during active learning:
#   Early iterations: 0.50 → Mid iterations: 0.60 → Final iterations: 0.70
# The value below reflects the final threshold used.
CONFIDENCE_THRESHOLD = 0.7
EVAL_THRESHOLD = 20
BATCH_SIZE = 1000
DYNAMIC_EVAL_EVERY_N = 3

# ── Synthetic augmentation (size-based only) ─────────────────────────────────
# NOTE: These are the final-iteration settings. Earlier iterations used more
# aggressive counts and lower class-size thresholds, relaxed as training grew.
USE_LLM = True
SMALL_CLASS_THRESHOLD = 100
LARGE_CLASS_THRESHOLD = 250
SMALL_CLASS_AUGMENTATIONS = 20
MEDIUM_CLASS_AUGMENTATIONS = 15
LARGE_CLASS_AUGMENTATIONS = 10
OTHER_AUGMENTATIONS = 2

# ── Canonical label set ──────────────────────────────────────────────────────
ALLOWED_LABELS = [
    "Active Transport Environment",
    "Air Quality",
    "NCD's",
    "Other",
    "Recreational PA",
    "Recreational and Sports Environment",
    "Total PA",
    "Transport PA",
]

# ── File paths ───────────────────────────────────────────────────────────────
AUGMENTED_DATA_PATH = os.path.join(DATA_DIR, "docs_and_labels_augmented.xlsx")
STATIC_HOLDOUT_PATH = os.path.join(DATA_DIR, "holdout_static.xlsx")
DYNAMIC_HOLDOUT_PATH = os.path.join(DATA_DIR, "holdout_dynamic.xlsx")
CLF_PATH = os.path.join(DATA_DIR, "classifier.pkl")
LABEL_MAPS_PATH = os.path.join(DATA_DIR, "label_maps.pkl")
PREDICTIONS_OUT = os.path.join(DATA_DIR, "all_predictions.xlsx")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")

# ── Embedding model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME_OR_PATH = "sentence-transformers/all-mpnet-base-v2"
EMBED_BATCH_SIZE = 64

# ── SVM ──────────────────────────────────────────────────────────────────────
SVM_TEMPLATE = SVC(
    probability=True, kernel="rbf", C=100.0,
    gamma="scale", class_weight="balanced",
)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("citymove")


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
    "total pa": "Total PA",          # kept as separate category
    "other": "Other",
    "active transport environment": "Active Transport Environment",
}


def normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return label
    return CANONICAL_MAP.get(label.strip().lower(), label.strip())


def validate_labels(labels: pd.Series, allowed: Optional[List[str]] = None) -> List[str]:
    unique = sorted(labels.astype(str).map(normalize_label).unique().tolist())
    if allowed is not None:
        bad = set(unique) - set(allowed)
        if bad:
            raise ValueError(f"Unknown labels: {sorted(bad)}")
        return sorted(allowed)
    return unique


def build_label_maps(labels_unique):
    l2i = {lab: i for i, lab in enumerate(labels_unique)}
    i2l = {i: lab for lab, i in l2i.items()}
    return l2i, i2l


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text_for_dedup(text) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


# =============================================================================
# BOILERPLATE STRIPPING (exploratory — may not be active in final run)
# =============================================================================

MIN_TITLE_LENGTH = 15


def extract_discriminative_text(text: str) -> str:
    """
    Extract the discriminative portion of a dataset description.

    Many docs follow "Title: long boilerplate description..." where
    the title carries the class signal and the description is a shared
    survey template. This extracts just the title when it's informative
    enough (≥ MIN_TITLE_LENGTH chars), otherwise keeps the full text.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    idx = text.find(":")
    if idx >= MIN_TITLE_LENGTH:
        return text[:idx].strip()
    return text.strip()


# =============================================================================
# EMBEDDING CACHE
# =============================================================================

@dataclass
class EmbeddingCache:
    model: SentenceTransformer
    cache: Dict[str, np.ndarray] = field(default_factory=dict)

    def encode(self, texts: List[str]) -> np.ndarray:
        texts = [str(t) if not isinstance(t, str) else t for t in texts]
        stripped = [extract_discriminative_text(t) for t in texts]
        keys = [normalize_text_for_dedup(s) for s in stripped]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        if missing:
            embs = self.model.encode(
                [stripped[i] for i in missing],
                batch_size=EMBED_BATCH_SIZE, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=False,
            )
            for i, emb in zip(missing, embs):
                self.cache[keys[i]] = emb.astype(np.float32, copy=False)
        return np.vstack([self.cache[k] for k in keys])


# =============================================================================
# AUGMENTATION (size-based only)
# =============================================================================

def get_augmentation_count(label: str, label_counts: Dict[str, int]) -> int:
    """How many synthetic variants to generate for this label."""
    if label.lower() == "other":
        return OTHER_AUGMENTATIONS
    count = label_counts.get(label, 0)
    if count < SMALL_CLASS_THRESHOLD:
        return SMALL_CLASS_AUGMENTATIONS
    if count > LARGE_CLASS_THRESHOLD:
        return LARGE_CLASS_AUGMENTATIONS
    return MEDIUM_CLASS_AUGMENTATIONS


# =============================================================================
# MODEL
# =============================================================================

def train_svm(X, y, template=SVM_TEMPLATE):
    clf = clone(template)
    clf.fit(X, y)
    return clf


def predict_with_probs(clf, X, idx_to_label):
    probs = clf.predict_proba(X)
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [idx_to_label[int(i)] for i in pred_idx]
    return probs, pred_idx, pred_labels


# =============================================================================
# BATCHING
# =============================================================================

def create_shuffled_batches(df, batch_size):
    shuffled = df.sample(frac=1).reset_index(drop=True)
    return [
        shuffled.iloc[i: i + batch_size].reset_index(drop=True)
        for i in range(0, len(shuffled), batch_size)
    ]


# =============================================================================
# PROVENANCE
# =============================================================================

def ensure_provenance_columns(df):
    df = df.copy()
    if IS_SYNTHETIC_COL not in df.columns:
        df[IS_SYNTHETIC_COL] = False
    else:
        df[IS_SYNTHETIC_COL] = df[IS_SYNTHETIC_COL].fillna(False).astype(bool)
    if DOC_GROUP_COL not in df.columns:
        df[DOC_GROUP_COL] = [f"legacy_{i}" for i in range(len(df))]
    return df


# =============================================================================
# BACKUP / SAVE
# =============================================================================

def create_backup():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in [AUGMENTED_DATA_PATH, STATIC_HOLDOUT_PATH, CLF_PATH, LABEL_MAPS_PATH]:
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(BACKUP_DIR, f"{ts}_{os.path.basename(path)}"))
    logger.info("✓ Backup: %s", ts)


def save_progress(clf, train_df, static_holdout_df):
    create_backup()
    joblib.dump(clf, CLF_PATH)
    # Save label maps so downstream scripts can recover label ↔ index mapping
    labels_unique = sorted(train_df[LABEL_COL].astype(str).map(normalize_label).unique().tolist())
    l2i = {lab: i for i, lab in enumerate(labels_unique)}
    i2l = {i: lab for lab, i in l2i.items()}
    joblib.dump({"label_to_idx": l2i, "idx_to_label": i2l, "labels_unique": labels_unique}, LABEL_MAPS_PATH)
    train_df.to_excel(AUGMENTED_DATA_PATH, index=False)
    static_holdout_df.to_excel(STATIC_HOLDOUT_PATH, index=False)
    n_syn = int(train_df[IS_SYNTHETIC_COL].sum()) if IS_SYNTHETIC_COL in train_df.columns else 0
    logger.info("✓ Saved: %d train (%d synthetic), %d holdout",
                len(train_df), n_syn, len(static_holdout_df))


# =============================================================================
# LOAD / INITIALIZE  (two clear paths)
# =============================================================================

def _load_existing(emb_cache):
    """Load from previously saved augmented data + classifier."""
    logger.info("Loading: %s", AUGMENTED_DATA_PATH)
    train_df = pd.read_excel(AUGMENTED_DATA_PATH)
    train_df[LABEL_COL] = train_df[LABEL_COL].astype(str).map(normalize_label)
    train_df = ensure_provenance_columns(train_df).reset_index(drop=True)
    static_holdout_df = _load_static_holdout()
    l2i, i2l = build_label_maps(validate_labels(train_df[LABEL_COL], ALLOWED_LABELS))
    logger.info("Loaded: %d train, %d holdout", len(train_df), len(static_holdout_df))
    clf = joblib.load(CLF_PATH) if os.path.exists(CLF_PATH) else _fit_and_save(emb_cache, train_df, static_holdout_df, l2i)
    return clf, train_df, static_holdout_df, l2i, i2l


def _load_static_holdout():
    if os.path.exists(STATIC_HOLDOUT_PATH):
        df = pd.read_excel(STATIC_HOLDOUT_PATH)
        if len(df) > 0:
            df[LABEL_COL] = df[LABEL_COL].astype(str).map(normalize_label)
        return df.reset_index(drop=True)
    logger.warning("Static holdout not found — creating empty.")
    return pd.DataFrame(columns=[DOC_COL, LABEL_COL])


def _load_base_file():
    """Read and normalize the base training file."""
    if not os.path.exists(BASE_TRAIN_FILE):
        raise FileNotFoundError(f"Base file not found: {BASE_TRAIN_FILE}")
    df = pd.read_excel(BASE_TRAIN_FILE)
    df = df.rename(columns={BASE_DOCS_COL: DOC_COL, BASE_LABELS_COL: LABEL_COL})
    df = df[[DOC_COL, LABEL_COL]].copy()
    df[DOC_COL] = df[DOC_COL].astype(str).fillna("None")
    df[LABEL_COL] = df[LABEL_COL].astype(str).map(normalize_label)
    return df


def _cold_start(emb_cache):
    """Initialize from the base training file."""
    logger.info("Cold start from: %s", BASE_TRAIN_FILE)
    base_df = _load_base_file()
    train_df, static_holdout_df = _stratified_split(base_df)
    train_df[IS_SYNTHETIC_COL] = False
    train_df[DOC_GROUP_COL] = [f"base_{i}" for i in range(len(train_df))]
    logger.info("Split: %d train, %d holdout", len(train_df), len(static_holdout_df))
    l2i, i2l = build_label_maps(validate_labels(train_df[LABEL_COL], ALLOWED_LABELS))
    clf = _fit_and_save(emb_cache, train_df, static_holdout_df, l2i)
    return clf, train_df, static_holdout_df, l2i, i2l


def _stratified_split(df, test_size=0.2):
    try:
        tr, ho = train_test_split(df, test_size=test_size, random_state=42, stratify=df[LABEL_COL])
    except ValueError:
        logger.warning("Stratified split failed — falling back to random.")
        tr, ho = train_test_split(df, test_size=test_size, random_state=42)
    return tr.reset_index(drop=True), ho.reset_index(drop=True)


def _fit_and_save(emb_cache, train_df, static_holdout_df, l2i):
    X = emb_cache.encode(train_df[DOC_COL].tolist())
    y = train_df[LABEL_COL].map(l2i).astype(int).to_numpy()
    clf = train_svm(X, y)
    save_progress(clf, train_df, static_holdout_df)
    logger.info("✓ Trained SVM on %d docs", len(train_df))
    return clf


def load_progress_or_initialize(emb_cache):
    if os.path.exists(AUGMENTED_DATA_PATH):
        return _load_existing(emb_cache)
    return _cold_start(emb_cache)


# =============================================================================
# EVALUATION
# =============================================================================

def _filter_to_known_labels(holdout_df, l2i):
    """Drop holdout rows with labels not in classifier's label map."""
    known = set(l2i.keys())
    holdout_df = holdout_df.copy()
    holdout_df[LABEL_COL] = holdout_df[LABEL_COL].astype(str).map(normalize_label)
    missing = set(holdout_df[LABEL_COL].unique()) - known
    if missing:
        logger.warning("Dropping unknown holdout labels: %s", sorted(missing))
        holdout_df = holdout_df[holdout_df[LABEL_COL].isin(known)].reset_index(drop=True)
    return holdout_df


def _compute_metrics(y_true, y_pred, i2l):
    present = sorted(set(y_true) | set(y_pred))
    names = [i2l[i] for i in present]
    report = classification_report(y_true, y_pred, labels=present, target_names=names, zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"f1_weighted": f1_w, "f1_macro": f1_m, "report": report}


def evaluate_on_holdout(holdout_df, clf, emb_cache, l2i, i2l, name="Holdout"):
    if len(holdout_df) == 0:
        return {}
    holdout_df = _filter_to_known_labels(holdout_df, l2i)
    if len(holdout_df) == 0:
        return {}
    X = emb_cache.encode(holdout_df[DOC_COL].tolist())
    y_true = holdout_df[LABEL_COL].map(l2i).astype(int).to_numpy()
    _, pred_idx, _ = predict_with_probs(clf, X, i2l)
    result = _compute_metrics(y_true, pred_idx.astype(int), i2l)
    logger.info("%s: F1w=%.4f | F1m=%.4f\n%s", name, result["f1_weighted"], result["f1_macro"], result["report"])
    return result


def _split_groups_for_dynamic_holdout(train_df, frac=0.2):
    """Split doc_group_ids into train/holdout sets (stratified)."""
    originals = train_df[~train_df[IS_SYNTHETIC_COL]]
    group_labels = originals.groupby(DOC_GROUP_COL)[LABEL_COL].first()
    try:
        tr, ho = train_test_split(
            group_labels.index.tolist(), test_size=frac,
            random_state=42, stratify=group_labels.values.tolist(),
        )
    except ValueError:
        tr, ho = train_test_split(group_labels.index.tolist(), test_size=frac, random_state=42)
    return set(tr), set(ho)


def _build_dynamic_splits(train_df, train_groups, holdout_groups):
    train_split = train_df[train_df[DOC_GROUP_COL].isin(train_groups)].reset_index(drop=True)
    holdout_split = train_df[
        train_df[DOC_GROUP_COL].isin(holdout_groups) & ~train_df[IS_SYNTHETIC_COL]
    ].reset_index(drop=True)
    return train_split, holdout_split


def evaluate_dynamic_holdout(train_df, emb_cache, l2i, i2l, frac=0.2):
    if DOC_GROUP_COL not in train_df.columns:
        return None
    if train_df[~train_df[IS_SYNTHETIC_COL]].shape[0] < 10:
        return None
    train_groups, holdout_groups = _split_groups_for_dynamic_holdout(train_df, frac)
    train_split, holdout_split = _build_dynamic_splits(train_df, train_groups, holdout_groups)
    if len(holdout_split) < 5:
        return None
    X_tr = emb_cache.encode(train_split[DOC_COL].tolist())
    y_tr = train_split[LABEL_COL].map(l2i).astype(int).to_numpy()
    temp_clf = train_svm(X_tr, y_tr)
    evaluate_on_holdout(holdout_split, temp_clf, emb_cache, l2i, i2l, "Dynamic Holdout")
    holdout_split.to_excel(DYNAMIC_HOLDOUT_PATH, index=False)
    return holdout_split


# =============================================================================
# REVIEW LOOP
# =============================================================================

ReviewedItem = namedtuple("ReviewedItem", ["doc", "label", "predicted", "margin", "was_correct"])


def _display_item(batch_idx, n_batches, rank, conf, margin, pred_label, second_label, second_prob, doc):
    print(f"\n\n{'=' * 92}")
    print(f"Batch {batch_idx}/{n_batches} | Review #{rank} | "
          f"conf={conf:.3f} | margin={margin:.3f} | predicted={pred_label}")
    print(f"  2nd-best: {second_label} (prob={second_prob:.3f})")
    print(f"{'-' * 92}")
    print(doc)
    print(f"{'=' * 92}\n")


def _ask_label(pred_label, second_label, second_prob):
    """Ask user for the correct label. Returns (label, was_correct) or ('EXIT', False)."""
    ans = input("Correct? [Y/N/EXIT]: ").strip().lower()
    if ans == "exit":
        return "EXIT", False
    if ans == "y":
        return normalize_label(pred_label), True

    print(f"\nSecond-best: {second_label} (prob={second_prob:.3f})")
    if input("Use second-best? [Y/N]: ").strip().lower() == "y":
        return normalize_label(second_label), False
    return normalize_label(input("Enter correct label: ").strip()), False


def _review_single_item(i, texts, probs, pred_labels, i2l, batch_idx, n_batches, rank, margin):
    """Present one item for review. Returns ReviewedItem, 'SKIP', or 'EXIT'."""
    second_idx = int(np.argsort(probs[i])[-2])
    second_label = i2l[second_idx]
    _display_item(batch_idx, n_batches, rank, float(probs[i].max()), margin,
                  pred_labels[i], second_label, probs[i, second_idx], texts[i])
    label, was_correct = _ask_label(pred_labels[i], second_label, probs[i, second_idx])
    if label == "EXIT":
        return "EXIT"
    if label not in ALLOWED_LABELS:
        print(f"ERROR: '{label}' not in {ALLOWED_LABELS}")
        return "SKIP"
    return ReviewedItem(texts[i], label, pred_labels[i], margin, was_correct)


def review_batch(texts, probs, pred_labels, i2l, batch_idx, n_batches):
    """Present low-confidence items for review. Returns list or None (EXIT)."""
    max_conf = probs.max(axis=1)
    order = np.argsort(max_conf)
    margins = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
    reviewed = []
    for rank, i in enumerate(order, start=1):
        if float(max_conf[i]) >= CONFIDENCE_THRESHOLD:
            logger.info("Confidence ≥ %.2f at rank %d. Stopping.", CONFIDENCE_THRESHOLD, rank)
            break
        if len(reviewed) >= EVAL_THRESHOLD:
            break
        result = _review_single_item(i, texts, probs, pred_labels, i2l, batch_idx, n_batches, rank, float(margins[i]))
        if result == "EXIT":
            return None
        if result != "SKIP":
            reviewed.append(result)
    return reviewed


# =============================================================================
# AUGMENTATION + DEDUP
# =============================================================================

def augment_reviewed_items(reviewed, train_df, seen_texts):
    """
    For each reviewed item: add original + synthetic variants to training data.

    Returns list of new row dicts.
    """
    label_counts = train_df[LABEL_COL].value_counts().to_dict()
    new_rows = []

    for r_idx, item in enumerate(reviewed):
        group_id = f"rev_R{r_idx}"
        _add_original(item.doc, item.label, group_id, seen_texts, new_rows)
        if USE_LLM:
            num_aug = get_augmentation_count(item.label, label_counts)
            _add_synthetic(item.doc, item.label, group_id, num_aug, seen_texts, new_rows)

    return new_rows


def _add_original(doc, label, group_id, seen_texts, new_rows):
    key = normalize_text_for_dedup(doc)
    if key not in seen_texts:
        new_rows.append({DOC_COL: doc, LABEL_COL: label, IS_SYNTHETIC_COL: False, DOC_GROUP_COL: group_id})
        seen_texts.add(key)


def _add_synthetic(doc, label, group_id, num_aug, seen_texts, new_rows):
    try:
        synth_docs, _ = generate_synthetic_data(doc=doc, label=label, num_augmentations=num_aug)
    except Exception as exc:
        logger.error("Synthetic generation failed for '%s': %s", label, exc)
        return

    for sd in synth_docs:
        key = normalize_text_for_dedup(sd)
        if key not in seen_texts:
            new_rows.append({DOC_COL: sd, LABEL_COL: label, IS_SYNTHETIC_COL: True, DOC_GROUP_COL: group_id})
            seen_texts.add(key)


# =============================================================================
# MAIN
# =============================================================================

def _log_confidence(probs, pred_labels):
    """Log a one-line confidence summary for the current batch."""
    mc = probs.max(axis=1)
    n_below = int((mc < CONFIDENCE_THRESHOLD).sum())
    logger.info(
        "Confidence: %d/%d below %.2f | median=%.3f | q10=%.3f | q90=%.3f",
        n_below, len(mc), CONFIDENCE_THRESHOLD,
        float(np.median(mc)), float(np.quantile(mc, 0.1)), float(np.quantile(mc, 0.9)),
    )


def _retrain(emb_cache, train_df):
    """Retrain SVM on current training data. Returns (clf, l2i, i2l)."""
    labels_unique = validate_labels(train_df[LABEL_COL], ALLOWED_LABELS)
    l2i, i2l = build_label_maps(labels_unique)
    train_df[DOC_COL] = train_df[DOC_COL].fillna("").astype(str)
    X = emb_cache.encode(train_df[DOC_COL].tolist())
    y = train_df[LABEL_COL].map(l2i).astype(int).to_numpy()
    clf = train_svm(X, y)
    logger.info("✓ Retrained SVM on %d docs", len(train_df))
    return clf, l2i, i2l


def _init_dedup_registry(train_df, static_holdout_df):
    seen = set()
    for src in [train_df[DOC_COL], static_holdout_df.get(DOC_COL, pd.Series(dtype=str))]:
        seen.update(normalize_text_for_dedup(t) for t in src.tolist())
    logger.info("Dedup registry: %d unique texts", len(seen))
    return seen


def _generate_final_predictions(clf, emb_cache, i2l, target_df):
    logger.info("GENERATING PREDICTIONS FOR TARGET FILE")
    X = emb_cache.encode(target_df[TEXT_COL].astype(str).tolist())
    probs, _, pred_labels = predict_with_probs(clf, X, i2l)
    target_df["Predicted_label"] = pred_labels
    target_df["Predicted_confidence"] = probs.max(axis=1)
    target_df.to_excel(PREDICTIONS_OUT, index=False)
    logger.info("✓ Saved: %s (%d docs)", PREDICTIONS_OUT, len(target_df))


def _review_and_augment(batch, batch_idx, n_batches, clf, emb_cache, i2l, train_df, seen_texts):
    """Review batch → augment. Returns new_rows list, or None on EXIT, or [] if nothing reviewed."""
    texts = batch[TEXT_COL].astype(str).tolist()
    probs, _, pred_labels = predict_with_probs(clf, emb_cache.encode(texts), i2l)
    _log_confidence(probs, pred_labels)
    reviewed = review_batch(texts, probs, pred_labels, i2l, batch_idx, n_batches)
    if reviewed is None:
        return None
    if not reviewed:
        return []
    return augment_reviewed_items(reviewed, train_df, seen_texts)


def _process_batch(batch, batch_idx, n_batches, clf, emb_cache, i2l, l2i, train_df, seen_texts, static_holdout_df):
    """Process one batch. Returns (clf, train_df, l2i, i2l) or None on EXIT."""
    new_rows = _review_and_augment(batch, batch_idx, n_batches, clf, emb_cache, i2l, train_df, seen_texts)
    if new_rows is None:
        save_progress(clf, train_df, static_holdout_df)
        return None
    if new_rows:
        train_df = pd.concat([train_df, pd.DataFrame(new_rows)], ignore_index=True)
        clf, l2i, i2l = _retrain(emb_cache, train_df)
    if batch_idx % DYNAMIC_EVAL_EVERY_N == 0:
        evaluate_dynamic_holdout(train_df, emb_cache, l2i, i2l)
    save_progress(clf, train_df, static_holdout_df)
    return clf, train_df, l2i, i2l


def _setup():
    """Load models, data, target file. Returns (emb_cache, clf, train_df, static_holdout_df, l2i, i2l, seen, target_df, batches)."""
    emb_cache = EmbeddingCache(model=SentenceTransformer(EMBEDDING_MODEL_NAME_OR_PATH))
    clf, train_df, static_holdout_df, l2i, i2l = load_progress_or_initialize(emb_cache)
    seen_texts = _init_dedup_registry(train_df, static_holdout_df)
    target_df = pd.read_excel(TARGET_FILE)
    target_df[TEXT_COL] = target_df[TEXT_COL].astype(str).fillna("None")
    batches = create_shuffled_batches(target_df, BATCH_SIZE)
    logger.info("Created %d batches of ~%d docs", len(batches), BATCH_SIZE)
    return emb_cache, clf, train_df, static_holdout_df, l2i, i2l, seen_texts, target_df, batches


def main():
    logger.info("CITY-MOVE CLASSIFICATION — v8.0")
    emb_cache, clf, train_df, static_holdout_df, l2i, i2l, seen_texts, target_df, batches = _setup()

    for batch_idx, batch in enumerate(batches, start=1):
        logger.info("BATCH %d / %d  (n=%d)", batch_idx, len(batches), len(batch))
        result = _process_batch(batch, batch_idx, len(batches), clf, emb_cache, i2l, l2i, train_df, seen_texts, static_holdout_df)
        if result is None:
            logger.info("Exit requested.")
            return
        clf, train_df, l2i, i2l = result

    evaluate_on_holdout(static_holdout_df, clf, emb_cache, l2i, i2l, "Static Holdout")
    evaluate_dynamic_holdout(train_df, emb_cache, l2i, i2l)
    _generate_final_predictions(clf, emb_cache, i2l, target_df)
    logger.info("COMPLETE!")


# =============================================================================
# POST-PROCESSING UTILITIES
# =============================================================================

def get_label_maps_from_saved_data():
    train_df = pd.read_excel(AUGMENTED_DATA_PATH)
    train_df[LABEL_COL] = train_df[LABEL_COL].fillna("Other").astype(str).map(normalize_label)
    labels_unique = sorted(train_df[LABEL_COL].unique().tolist())
    l2i, i2l = build_label_maps(labels_unique)
    return l2i, i2l, labels_unique


def predict_unseen_file(input_path, output_path, text_col=TEXT_COL):
    clf = joblib.load(CLF_PATH)
    _, i2l, _ = get_label_maps_from_saved_data()
    df = pd.read_csv(input_path) if input_path.lower().endswith(".csv") else pd.read_excel(input_path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not in {input_path}")
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME_OR_PATH)
    X = emb_model.encode(df[text_col].astype(str).fillna("None").tolist(),
                         batch_size=EMBED_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    probs = clf.predict_proba(X)
    df["Predicted_label"] = [i2l[int(i)] for i in np.argmax(probs, axis=1)]
    df["Predicted_confidence"] = probs.max(axis=1)
    df.to_excel(output_path, index=False)
    logger.info("✓ Saved: %s", output_path)


# =============================================================================

if __name__ == "__main__":
    main()
