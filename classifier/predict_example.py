"""Label dataset descriptions with the published City-Move classifier.

    python classifier/predict_example.py "Number of cyclists counted per street segment"
    python classifier/predict_example.py --file catalogue.csv --column description --out labelled.csv

Each text is embedded in full with all-mpnet-base-v2 (unnormalised, as in
training) and scored by the SVM. The label is the class with the highest
probability. Predictions below 0.70, the final review threshold of the active
learning loop, should be checked by hand; the paper used the classifier to
assist screening, not to decide inclusion. Texts should be in English.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).resolve().parent / "data"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
REVIEW_THRESHOLD = 0.70


def load():
    clf = joblib.load(DATA / "classifier.pkl")
    idx_to_label = json.loads((DATA / "label_maps.json").read_text(encoding="utf-8"))["idx_to_label"]
    labels = [idx_to_label[str(int(c))] for c in clf.classes_]
    return clf, labels, SentenceTransformer(EMBEDDING_MODEL)


def predict(texts, clf, labels, embedder):
    texts = [str(t) for t in texts]
    X = embedder.encode(texts, batch_size=32, show_progress_bar=len(texts) > 200, normalize_embeddings=False)
    proba = clf.predict_proba(X)
    order = np.argsort(-proba, axis=1)
    rows = np.arange(len(texts))
    out = pd.DataFrame({
        "text": texts,
        "label": [labels[i] for i in order[:, 0]],
        "confidence": proba[rows, order[:, 0]].round(3),
        "second_label": [labels[i] for i in order[:, 1]],
        "second_confidence": proba[rows, order[:, 1]].round(3),
    })
    out["needs_review"] = out["confidence"] < REVIEW_THRESHOLD
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texts", nargs="*", help="one or more descriptions to label")
    ap.add_argument("--file", help="CSV or XLSX file with descriptions")
    ap.add_argument("--column", help="column holding the descriptions (with --file)")
    ap.add_argument("--out", help="write results to this CSV instead of printing them")
    args = ap.parse_args()

    if args.file:
        path = Path(args.file)
        df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
        if args.column not in df.columns:
            ap.error(f"--column must be one of: {', '.join(map(str, df.columns))}")
        texts = df[args.column].fillna("").astype(str).tolist()
    elif args.texts:
        texts = args.texts
    else:
        ap.error("give texts or --file")

    result = predict(texts, *load())
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"wrote {len(result)} rows to {args.out}; {int(result['needs_review'].sum())} below {REVIEW_THRESHOLD}")
    else:
        with pd.option_context("display.max_colwidth", 60, "display.width", 160):
            print(result.to_string(index=False))


if __name__ == "__main__":
    main()
