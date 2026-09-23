# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 13:21:42 2026

@author: Oberon De Deurwaerder
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path

# -*- coding: utf-8 -*-
"""
Reproducible sampling: draw N rows from each DataFrame listed in DF_SOURCES,
select a specified text column from each, and combine into docs_df.

- One master RANDOM_SEED controls everything
- Per-DF seeds isolate randomness per DataFrame
- Enforces: each DF must have at least N_PER_DF rows (no "small df" handling)
"""

# =========================
# CONFIG
# =========================
RANDOM_SEED: int = 92
N_PER_DF: int = 100

# =========================
# DATAFRAMES TO SAMPLE FROM
# =========================
# Folder with the translated catalogue exports (not part of this repository;
# they will be deposited with the analysis files). Set CITYMOVE_CITY_DATA.
import os
DATA_DIR = os.environ.get("CITYMOVE_CITY_DATA", "City data to merge") + os.sep

df_a = pd.read_excel(DATA_DIR + "SiStat_api_data.xlsx") #Ljubljana
df_b = pd.read_excel(DATA_DIR + "UBOS Data scrape.xlsx") # Kampala
df_c = pd.read_excel(DATA_DIR + "antwerp_new_df_translated_full.xlsx") # Antwerp
df_d = pd.read_excel(DATA_DIR + "Bogota_final_df_with_translations.xlsx") # Bogota
df_e = pd.read_excel(DATA_DIR + "Lima_full_translated_df_23012026_final.xlsx") # Lima
df_f = pd.read_excel(DATA_DIR + "rotterdam_final_translations_2024.xlsx") # Rotterdam
df_g = pd.read_excel(DATA_DIR + "geo_data_translated.xlsx") # Geo-data Antwerpen
df_h = pd.read_excel(DATA_DIR + "Bogota_movilidad_df_translations.xlsx") # Bogota Mobility Directorate Data
df_i = pd.read_excel(DATA_DIR + "rotterdam_new_indicators_2026_translated_UPDATED.xlsx") # New indicators Rotterdam
df_j = pd.read_excel(DATA_DIR + "como_vamos_processed.xlsx") # Bogota Como Vamos

# Each tuple: (df_variable_name_in_IDE, column_name_to_extract_as_doc)
DF_SOURCES: List[Tuple[str, str]] = [
     ("df_a", "path"),
     ("df_b", "name"),
     ("df_c", "Translated_text"),
     ("df_d", "Translated_text"),
     ("df_e", "Translated_text"),
     ("df_f", "Translated_text")
]

# =========================
# CORE
# =========================
def _resolve_df(df_name: str, namespace: Dict) -> pd.DataFrame:
    if df_name not in namespace:
        raise KeyError(f"DataFrame '{df_name}' not found in current namespace (globals()).")
    obj = namespace[df_name]
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Object '{df_name}' exists but is not a pandas DataFrame (type={type(obj)}).")
    return obj


def build_docs_df(
    df_sources: List[Tuple[str, str]],
    n_per_df: int,
    random_seed: int,
    namespace: Dict,
    dropna_docs: bool = True,
    doc_colname: str = "doc",
) -> pd.DataFrame:
    """
    For each (df_name, text_col) in df_sources:
      - sample n_per_df rows reproducibly
      - extract df[text_col] into docs_df[doc_colname]
      - add provenance columns

    Deterministic given:
      - same seed
      - same df order in df_sources
      - same df contents and row order
    """

    if not df_sources:
        raise ValueError("df_sources is empty. Provide at least one (df_name, column_name) pair.")

    master_rng = np.random.default_rng(random_seed)
    parts = []

    for df_name, text_col in df_sources:
        df = _resolve_df(df_name, namespace)

        if text_col not in df.columns:
            raise KeyError(
                f"Column '{text_col}' not found in DataFrame '{df_name}'. "
                f"Available columns: {list(df.columns)[:25]}{'...' if len(df.columns) > 25 else ''}"
            )

        n_available = len(df)
        if n_available < n_per_df:
            raise ValueError(
                f"DataFrame '{df_name}' has {n_available} rows, but n_per_df={n_per_df}. "
                "Increase data or lower N_PER_DF."
            )

        # Per-DF deterministic seed derived from master seed
        per_df_seed = int(master_rng.integers(0, 2**32 - 1))
        per_df_rng = np.random.default_rng(per_df_seed)

        chosen_pos = per_df_rng.choice(n_available, size=n_per_df, replace=False)

        # Build a standardized output slice
        sampled = df.iloc[chosen_pos].copy()

        out = pd.DataFrame({
            doc_colname: sampled[text_col].astype("string"),
            "_source_df": df_name,
            "_source_col": text_col,
            "_source_row_index": sampled.index.to_numpy(),
            "_per_df_seed": per_df_seed,
        })
        # Stable row id for anti-join later
        out["_row_uid"] = out["_source_df"].astype(str) + "::" + out["_source_row_index"].astype(str)
        
        if dropna_docs:
            out = out[out[doc_colname].notna() & (out[doc_colname].str.len() > 0)]

        parts.append(out)

    docs_df = pd.concat(parts, ignore_index=True)
    docs_df["_sample_id"] = np.arange(len(docs_df), dtype=int)

    return docs_df


# =========================
# RUN
# =========================
docs_df = build_docs_df(
    df_sources=DF_SOURCES,
    n_per_df=N_PER_DF,
    random_seed=RANDOM_SEED,
    namespace=globals(),      # Spyder/IPython: grab DataFrames from your IDE namespace
    dropna_docs=True,
    doc_colname="doc",
)


print("docs_df shape:", docs_df.shape)
print(docs_df["_source_df"].value_counts())

docs_df.to_excel("sampled_docs_df.xlsx")

# LABEL COUNTS STARTING SET 

LABELED_PATH = Path(__file__).resolve().parents[1] / "data" / "sampled_docs_df_04022026.xlsx"

train_df = pd.read_excel(LABELED_PATH)

print("Loaded labeled df shape:", train_df.shape)
print("Columns:", list(train_df.columns))

LABEL_COL = "labels"  # adjust if needed

label_counts = (
    train_df[LABEL_COL]
    .value_counts(dropna=False)
    .rename_axis("label")
    .to_frame("n")
    .assign(pct=lambda x: (x["n"] / x["n"].sum()).round(4))
    .reset_index()
)

print(label_counts)

#%% 
# =========================
# CHAPTER: BUILD PREDICTION DF (ALL NON-SAMPLED ROWS)
# =========================

# Add more DFs here (and choose which column becomes "doc" for prediction)
# You can include df_a..df_f again if you want "everything else" from them too.
PRED_SOURCES: List[Tuple[str, str]] = [
    ("df_a", "path"),
    ("df_b", "name"),
    ("df_c", "Translated_text"),
    ("df_d", "Translated_text"),
    ("df_e", "Translated_text"),
    ("df_f", "Translated_text"),
    ("df_g", "Translated_text"),
    ("df_h", "Translated_text"),
    ("df_i", "Translated_text"),
    ("df_j", "Variable description")
]

def build_to_predict_df(
    pred_sources: List[Tuple[str, str]],
    sampled_docs_df: pd.DataFrame,
    namespace: Dict,
    doc_colname: str = "doc",
    dropna_docs: bool = True,
) -> pd.DataFrame:
    """
    Build a unified prediction dataframe consisting of ALL rows from pred_sources
    that are NOT present in sampled_docs_df (based on _row_uid).
    """

    if "_row_uid" not in sampled_docs_df.columns:
        raise KeyError("sampled_docs_df is missing '_row_uid'. Add it in the sampling chapter first.")

    sampled_uids = set(sampled_docs_df["_row_uid"].astype("string"))

    parts = []
    for df_name, text_col in pred_sources:
        df = _resolve_df(df_name, namespace)

        if text_col not in df.columns:
            raise KeyError(
                f"Column '{text_col}' not found in DataFrame '{df_name}'. "
                f"Available columns: {list(df.columns)[:25]}{'...' if len(df.columns) > 25 else ''}"
            )

        # Build row-level uid for ALL rows in this df
        row_index = df.index.to_numpy()
        row_uid = pd.Series(df_name, index=df.index, dtype="string") + "::" + pd.Series(row_index, index=df.index).astype("string")

        # Keep only rows not sampled
        mask_keep = ~row_uid.isin(sampled_uids)

        out = pd.DataFrame({
            doc_colname: df.loc[mask_keep, text_col].astype("string"),
            "_source_df": df_name,
            "_source_col": text_col,
            "_source_row_index": row_index[mask_keep.to_numpy()],
        })

        out["_row_uid"] = out["_source_df"].astype("string") + "::" + out["_source_row_index"].astype("string")

        if dropna_docs:
            out = out[out[doc_colname].notna() & (out[doc_colname].str.len() > 0)]

        parts.append(out)

    to_predict_df = pd.concat(parts, ignore_index=True)
    to_predict_df["_predict_id"] = np.arange(len(to_predict_df), dtype=int)

    return to_predict_df


to_predict_df = build_to_predict_df(
    pred_sources=PRED_SOURCES,
    sampled_docs_df=docs_df,
    namespace=globals(),
    doc_colname="doc",
    dropna_docs=True,
)

print("to_predict_df shape:", to_predict_df.shape)
print(to_predict_df["_source_df"].value_counts())

to_predict_df.to_excel("to_predict_df.xlsx", index=False)

#%% Quick analytics 

dfs = [df_a, df_b, df_c, df_d, df_e, df_f, df_g, df_h, df_i, df_j]
total_rows = sum(len(df) for df in dfs)

# Map names to actual DataFrame objects
df_map = {
    "df_a": df_a,
    "df_b": df_b,
    "df_c": df_c,
    "df_d": df_d,
    "df_e": df_e,
    "df_f": df_f,
    "df_g": df_g,
    "df_h": df_h,
    "df_i": df_i,
    "df_j": df_j
}

overview_df = (
    pd.DataFrame({
        "dataframe": df_map.keys(),
        "n_rows": [len(df) for df in df_map.values()]
    })
    .assign(pct_total=lambda x: (x["n_rows"] / x["n_rows"].sum()).round(4))
    .sort_values("n_rows", ascending=False)
    .reset_index(drop=True)
)

print(overview_df)

total_rows = overview_df["n_rows"].sum()
print("Total rows across df_a–df_i:", total_rows)

total_chars = sum(
    globals()[f"df_{c}"]["Translated_text"].astype("string").str.len().sum()
    for c in "abcdefghi"
)

