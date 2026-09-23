# City-Move WP4: routine data for physical activity in six cities

This repository accompanies the City-Move WP4 methods paper. It holds the master
inventory of routine data records identified for **Antwerp, Bogotá, Kampala, Lima,
Ljubljana and Rotterdam**, the indicator framework the records were mapped to, the
classifier used to screen the cities' data catalogues, and an interactive map for
exploring the records.

**Interactive map: <https://oberon-dd.github.io/citymove_data_map/>**

Click a city to see how its records spread over the framework's domains, open a domain
to see every indicator with its direct and proxy records (including the indicators with
none), and open an indicator to list the records themselves, each linked to its source.

## In numbers

| | |
|---|---|
| Records identified | 8,557 |
| Analysed inventory | 8,071 (3,990 direct matches, 4,081 relevant proxies) |
| Excluded after review | 486 |
| Framework | 51 indicators in 13 clusters and 7 domains (v5.1) |
| Catalogued records screened by the classifier | 65,022 |

## Contents

| Path | What it is |
|---|---|
| [`index.html`](index.html), [`map_data.js`](map_data.js) | The interactive map (GitHub Pages). `map_data.js` is generated from the master inventory. |
| [`data/`](data) | The master inventory as CSV and XLSX, the framework, and the [data dictionary](data/DATA_DICTIONARY.md). |
| [`classifier/`](classifier) | The SVM classifier of Appendix A: model, training and holdout data, scripts, outputs, and a [provenance note](classifier/PROVENANCE.md). |
| [`tools/`](tools) | Scripts that build the public data files and the map data. |

## Using the data

```python
import pandas as pd

master = pd.read_csv("data/WP4_master_full.csv", keep_default_na=False)
analysed = master[master["fit"] != "PROPOSED EXIT"]          # the 8,071 records of the paper
direct = analysed[analysed["fit"] == "Direct match"]
print(direct.groupby("city")["uid"].count())
```

Each record has a `resource_url` that leads to the dataset, layer, table or file on the
publisher's platform. The [data dictionary](data/DATA_DICTIONARY.md) explains every column.

## Rebuilding

The map reads `map_data.js`; regenerate it after any change to the master:

```bash
python tools/build_map.py
```

`tools/prepare_public_master.py` produced `data/WP4_master_full.*` from the internal
master, redacting maintainer names and e-mail addresses and editing the review notes for
publication (details in the data dictionary). Both scripts need `pandas` and `openpyxl`.
To apply the classifier to new descriptions, see [`classifier/README.md`](classifier/README.md).

## Data availability

The master inventory, the indicator framework, the classifier with its training and
holdout data, and the interactive map are openly available in this repository. The
catalogue exports, the per-city files, the full source tables of Appendix B and the
scripts that generate the paper's tables will be deposited with the analysis files and
linked here once they have a DOI.

## Planned additions

- The analysis files deposit described above.
- The scripts that harvested and translated the city catalogues, and the pipeline that
  characterised spatial resolution, retrospective depth and update cadence.
- Contributions of further cities to the map.

## Citation and licence

Please cite the paper (reference to follow on publication) and this repository
([`CITATION.cff`](CITATION.cff)). Data and code are released under
[CC0 1.0](LICENSE). Titles and descriptions are catalogue metadata from the publishers
named in the `source` column; the linked data remain under each publisher's terms.

City-Move is funded by the European Union's Horizon Europe programme.
