"""Build the public copy of the WP4 master inventory.

    python tools/prepare_public_master.py path/to/WP4_master_full.csv

Reads the internal master (8,557 rows x 24 columns) and writes
data/WP4_master_full.csv and data/WP4_master_full.xlsx. Rows, columns and all
classification fields are unchanged. Two things differ from the internal file:

1. Maintainer names and e-mail addresses that the portals embed in dataset
   descriptions are redacted (rules in tools/pii.py).
2. The review notes in `round_tag` are edited for publication: neutral wording
   ("after review"), line-wrap artefacts repaired, dashes normalised, and the
   workbook column letter replaced by the field name.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pii import collect_names, redact  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_CSV = REPO / "data" / "WP4_master_full.csv"
OUT_XLSX = REPO / "data" / "WP4_master_full.xlsx"
FRAMEWORK = REPO / "data" / "framework_v5_1.csv"

TEXT_COLUMNS = ["title", "description_en", "declared_spatial_coverage", "declared_temporal_coverage",
                "spatial_evidence", "temporal_depth_evidence", "cadence_evidence", "round_tag"]

# Review-note wording. Longest phrases first so the specific forms win.
NOTE_WORDING = [
    ("PROPOSED EXIT per your review — your review: exit (depots)", "PROPOSED EXIT after review: exit (depots)"),
    ("PROPOSED EXIT per your review — your review: MSI = cargo load, exit",
     "PROPOSED EXIT after review: MSI = cargo load, exit"),
    ("PROPOSED EXIT per your review — your review: exit", "PROPOSED EXIT after review: exit"),
    ("(your review) — your override: density only (GTFS)", "(after review): override, density only (GTFS)"),
    ("(your review) — your review: covers PM10", "(after review): covers PM10"),
    ("(your review)", "(after review)"),
    ("per your Antwerp trip-distance convention", "per the Antwerp trip-distance convention"),
    ("per your Antwerp swimming rows", "per the Antwerp swimming rows"),
    ("per your Antwerp convention", "per the Antwerp convention"),
    ("follow your Antwerp convention", "follow the Antwerp convention"),
    ("(your parking convention)", "(parking convention)"),
    ("(your zoning convention)", "(zoning convention)"),
    ("(your satisfaction-links-to-the-thing convention)", "(convention: satisfaction links to the thing rated)"),
    ("(manual addition, your sign-off)", "(manual addition, signed off after review)"),
    ("included per your decision for better data availability", "included after review for better data availability"),
    ("per your proxy call", "per the proxy call after review"),
    ("JUDGMENT: kept your Sports infrastructure and sports clubs (renamed); note siblings 4726/4728 carry the same "
     "content under Gym / fitness centre access - align whichever way you intend",
     "JUDGMENT: kept Sports infrastructure and sports clubs (renamed); siblings 4726/4728 carry the same content "
     "under Gym / fitness centre access, alignment left open"),
    ("kept as an unlinked proxy on Oberon's ruling", "kept as an unlinked proxy by ruling after review"),
    ("age band matches our record exactly", "age band matches this record exactly"),
    ("proxy link moved to column G", "proxy link moved to the proxy_link field"),
]

# Words glued together when the audit notes were unwrapped.
NOTE_WRAP_REPAIRS = [
    ("anotherrow", "another row"), ("anotherrecord", "another record"), ("asa dataset", "as a dataset"),
    ("andspan", "and span"), ("readsas", "reads as"), ("readas", "read as"), ("10December", "10 December"),
    ("surveillance(IRAS", "surveillance (IRAS"), ("burdencluster", "burden cluster"), ("SanMartin", "San Martin"),
    ("Tacna,Tumbes", "Tacna, Tumbes"), ("orabout", "or about"), ("NATIONALportal", "NATIONAL portal"),
    ("traffic,collections", "traffic, collections"), ("toll-unitregister", "toll-unit register"),
    ("highwaycrashes", "highway crashes"), ("perfamily", "per family"), ("theMunicipalidad", "the Municipalidad"),
    ("November2018", "November 2018"), ("'Harvestedfrom", "'Harvested from"), ("LastHarvest", "Last Harvest"),
    ("frequency,sequential", "frequency, sequential"), ("atdatosabiertos", "at datosabiertos"),
    ("formathtml", "format html"), ("andrejected", "and rejected"), ("aPeru exit", "a Peru exit"),
    ("RESOLVE ATALL", "RESOLVE AT ALL"), ("suchentries", "such entries"), ("thatcannot", "that cannot"),
    ("byopening", "by opening"), ("inmiraflores_ghosts", "in miraflores_ghosts"),
]


def clean_note(note):
    if not note:
        return note, 0
    before = note
    for old, new in NOTE_WORDING + NOTE_WRAP_REPAIRS:
        note = note.replace(old, new)
    note = note.replace(" — ", " - ").replace("—", "-")
    return note, int(note != before)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(src):
    m = pd.read_csv(src, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert m.shape == (8557, 24), m.shape
    names = collect_names(pd.concat([m[c] for c in TEXT_COLUMNS]))
    n_redact = n_cells = 0
    for col in TEXT_COLUMNS:
        out = []
        for v in m[col]:
            v2, k = redact(v, names)
            n_redact += k
            n_cells += int(k > 0)
            out.append(v2)
        m[col] = out
    notes = [clean_note(v) for v in m["round_tag"]]
    m["round_tag"] = [v for v, _ in notes]
    n_notes = sum(k for _, k in notes)

    leftover = m["round_tag"].str.contains(r"\byour\b|\byou\b|—", regex=True)
    assert not leftover.any(), m.loc[leftover, "round_tag"].unique()[:5]
    assert not pd.concat([m[c] for c in TEXT_COLUMNS]).str.contains("@").any()

    OUT_CSV.parent.mkdir(exist_ok=True)
    m.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", lineterminator="\n")

    fw = pd.read_csv(FRAMEWORK, dtype=str, keep_default_na=False)
    live = m[m["fit"] != "PROPOSED EXIT"]
    about = pd.DataFrame({"item": [
        "Dataset", "Rows", "Analysed inventory", "Excluded after review", "Columns", "Framework",
        "Redaction", "Review notes", "Licence", "Documentation"], "value": [
        "City-Move WP4 master inventory of routine data records for six cities",
        f"{len(m):,} records (sheet 'Master (full)')",
        f"{len(live):,} records where fit is 'Direct match' or 'Relevant proxy'",
        f"{(m['fit'] == 'PROPOSED EXIT').sum():,} records where fit is 'PROPOSED EXIT' (kept for transparency)",
        "See data/README.md in the repository",
        f"{len(fw)} indicators in 13 clusters and 7 domains (sheet 'Framework v5.1')",
        f"{n_redact} maintainer names or e-mail addresses in {n_cells} text cells replaced by '[redacted]'",
        f"{n_notes} review notes edited for publication (wording and line-wrap repairs only)",
        "CC0 1.0", "https://github.com/Oberon-DD/citymove_data_map"]})
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        m.to_excel(xw, sheet_name="Master (full)", index=False)
        fw.to_excel(xw, sheet_name="Framework v5.1", index=False)
        about.to_excel(xw, sheet_name="About", index=False)
        ws = xw.sheets["Master (full)"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        widths = {"uid": 16, "city": 11, "source": 30, "title": 45, "description_en": 70, "domain": 26,
                  "cluster": 28, "indicator_direct": 32, "fit": 15, "proxy_link": 30, "resource_url": 45,
                  "round_tag": 50}
        for i, col in enumerate(m.columns, start=1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = widths.get(col, 18)
        for name, w in (("Framework v5.1", (6, 28, 30, 38, 60, 16)), ("About", (22, 90))):
            for i, width in enumerate(w, start=1):
                xw.sheets[name].column_dimensions[xw.sheets[name].cell(1, i).column_letter].width = width

    print(f"input  {src}  sha256 {sha256(src)}")
    print(f"rows {len(m):,}  analysed {len(live):,}  excluded {(m['fit'] == 'PROPOSED EXIT').sum():,}")
    print("analysed per city:", live["city"].value_counts().to_dict())
    print(f"redactions {n_redact} in {n_cells} cells; review notes edited {n_notes}")
    for p in (OUT_CSV, OUT_XLSX):
        print(f"wrote {p.relative_to(REPO)}  sha256 {sha256(p)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
