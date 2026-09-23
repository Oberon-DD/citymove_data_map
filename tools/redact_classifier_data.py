"""Copy the classifier's training and holdout workbooks with maintainer names and
e-mail addresses redacted (rules in tools/pii.py).

    python tools/redact_classifier_data.py "path/to/PAPER SETUP FINAL" path/to/WP4_master_full.csv

Only the `docs` column changes. The model itself stores embeddings, not text,
so it is published unchanged; a redacted text no longer embeds to exactly the
vector the model was trained on (see classifier/PROVENANCE.md).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pii import collect_names, redact  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FILES = ["docs_and_labels_augmented.xlsx", "holdout_static.xlsx", "holdout_dynamic.xlsx",
         "sampled_docs_df_04022026.xlsx", "new_base_file_docs_and_labels.xlsx"]


def main(src_dir, master_csv):
    frames = {name: pd.read_excel(Path(src_dir) / name) for name in FILES}
    master = pd.read_csv(master_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    names = collect_names(list(master["description_en"]) + [t for df in frames.values() for t in df["docs"]])
    for name, df in frames.items():
        n_cells = n_subs = 0
        docs = []
        for v in df["docs"]:
            v2, k = redact(v, names)
            n_cells += int(k > 0)
            n_subs += k
            docs.append(v2)
        df["docs"] = docs
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": ""})
        out = REPO / "classifier" / "data" / name
        df.to_excel(out, index=False, sheet_name="Sheet1")
        print(f"{name}: {len(df)} rows, {n_subs} redactions in {n_cells} texts")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
