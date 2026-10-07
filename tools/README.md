# Build scripts

| Script | What it does |
|---|---|
| [`prepare_public_master.py`](prepare_public_master.py) | Writes `data/WP4_master_full.csv` and `.xlsx` from the internal master: redacts maintainer names and e-mail addresses and edits the review notes for publication. |
| [`check_links.py`](check_links.py) | Checks every record's link and writes `data/link_check.csv` (see the [data README](../data/README.md#link-check)). Takes a few minutes. |
| [`build_map.py`](build_map.py) | Writes `map_data.js`, the data behind the interactive map, from `data/WP4_master_full.csv` and `data/link_check.csv`, and points `index.html` at the new version. Records with a dead link get their platform's start page instead. |
| [`redact_classifier_data.py`](redact_classifier_data.py) | Copies the classifier's training and holdout workbooks with the same redactions. |
| [`pii.py`](pii.py) | The redaction rules shared by the scripts above. Names are collected from the source files at run time and never stored. |

All need `pandas` and `openpyxl`. After a change to the master, or now and then to
catch links that have died:

```bash
python tools/check_links.py
python tools/build_map.py
```

These scripts were developed with Claude (Anthropic) as a contributor.
