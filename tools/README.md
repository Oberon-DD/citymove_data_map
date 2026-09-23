# Build scripts

| Script | What it does |
|---|---|
| [`prepare_public_master.py`](prepare_public_master.py) | Writes `data/WP4_master_full.csv` and `.xlsx` from the internal master: redacts maintainer names and e-mail addresses and edits the review notes for publication. |
| [`build_map.py`](build_map.py) | Writes `map_data.js`, the data behind the interactive map, from `data/WP4_master_full.csv`. Rerun after every change to the master. |
| [`redact_classifier_data.py`](redact_classifier_data.py) | Copies the classifier's training and holdout workbooks with the same redactions. |
| [`pii.py`](pii.py) | The redaction rules shared by the scripts above. Names are collected from the source files at run time and never stored. |

All need `pandas` and `openpyxl`:

```bash
python tools/build_map.py
```
