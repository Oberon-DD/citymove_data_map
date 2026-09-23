# Master inventory: data dictionary

`WP4_master_full.csv` (UTF-8 with BOM) and `WP4_master_full.xlsx` hold the same
table: every routine data record identified for the six City-Move cities, with its
source, link, position in the indicator framework and spatial and temporal
characterisation. The workbook adds the framework (sheet *Framework v5.1*) and a
short description (sheet *About*).

| | |
|---|---|
| Rows | 8,557 records |
| Analysed inventory | 8,071 records: `fit` is `Direct match` (3,990) or `Relevant proxy` (4,081) |
| Excluded after review | 486 records: `fit` is `PROPOSED EXIT` (paper, Table B3) |
| Distinct identifiers in the analysed inventory | 8,056: 14 sources contribute more than one record (29 rows), each on a different indicator |
| Cities | Antwerp 4,953 · Bogotá 1,658 · Rotterdam 956 · Ljubljana 254 · Lima 214 · Kampala 36 (analysed records) |

To reproduce the paper's counts, drop the rows where `fit` is `PROPOSED EXIT`.

## Columns

| Column | Content |
|---|---|
| `uid` | Record identifier. `df_<x>::<n>` is entry *n* of the catalogue export tagged `df_<x>` (table below); `EXT:...` marks a record found beyond the catalogues (final search additions and manual additions). |
| `city` | Antwerp, Bogota, Kampala, Lima, Ljubljana or Rotterdam. |
| `source` | Platform or publisher. Catalogued platforms carry their pipeline tag in brackets. |
| `title` | Title as published, in the source language. |
| `description_en` | Description as published, translated into English where the source was not in English. |
| `domain` | Framework domain (7), see `framework_v5_1.csv`. |
| `cluster` | Framework cluster (13). |
| `indicator_direct` | For a direct match: the framework indicator the record operationalises. Empty when the record spans several indicators (see `spans`). |
| `fit` | `Direct match`: the record measures the indicator as operationalised. `Relevant proxy`: it bears on the indicator or cluster without measuring it. `PROPOSED EXIT`: excluded after review. |
| `proxy_link` | For a proxy: the indicator it is linked to. Empty for a contextual proxy, which counts at cluster level only. |
| `spans` | For a direct match covering several indicators: those indicators, separated by `;`. Such a record counts as direct evidence for each of them (paper, Table 3). |
| `spatial_label` | Finest spatial resolution: National, Regional, City-wide, Sub-city (district-zone), Sub-city (neighbourhood-block), Point-feature or Unknown. |
| `temporal_label` | Retrospective depth and update cadence, as `<depth>; <cadence>`. Depth: single year, 2-4, 5-9, 10-19 or >=20 years, unknown depth. Cadence: one-off, irregular, multi-year, annual-or-finer, unknown cadence. |
| `year_min`, `year_max` | First and last year covered, where determinable. |
| `year_span_elapsed` | `year_max` minus `year_min`. |
| `spatial_evidence` | How the spatial band was established (for example harvested platform metadata, a rule for the source system, parsing of the title, or probing the data file). |
| `temporal_depth_evidence`, `cadence_evidence` | How depth and cadence were established: HARVESTED (platform metadata), TEXT (parsed from title or description), AUTHORED (set by the reviewers from the source), UNKNOWN. |
| `resource_url` | Link to the record on the source platform: dataset page, layer, table or file. Where a viewer does not address layers separately, the viewer's landing page. |
| `declared_spatial_coverage`, `declared_temporal_coverage` | Coverage as stated by the publisher, recorded for records added beyond the catalogues. |
| `round_tag` | Review notes: why a record was added, amended, linked or excluded, with the date of the decision. File names in these notes refer to the analysis files deposit. |
| `source_row` | Row of the record in its city file in the analysis files deposit. `city` and `source_row` together identify a row of this table. |

## Catalogued platforms

| Tag | City | Platform (name used in the paper) |
|---|---|---|
| `df_a` | Ljubljana | SiStat (Statistical Office px API) |
| `df_b` | Kampala | Uganda Bureau of Statistics |
| `df_c` | Antwerp | Stad in Cijfers (indicator portal, Swing viewer) |
| `df_d` | Bogotá | Datos Abiertos Bogota |
| `df_e` | Lima | Plataforma Nacional de Datos Abiertos |
| `df_f` | Rotterdam | Onderzoek010 / Wijkprofiel (indicator portal, Swing viewer), 2024 export |
| `df_g` | Antwerp | Stad in Kaarten (geoportal) |
| `df_h` | Bogotá | ArcGIS Hub Movilidad (SIMUR) |
| `df_i` | Rotterdam | Onderzoek010 / Wijkprofiel (indicator portal, Swing viewer), 2026 export |
| `df_j` | Bogotá | Bogota Como Vamos (survey variable workbook) |
| `df_k` | Lima | Lima Como Vamos (survey variable workbook) |

## Differences from the internal master

The public file is produced from the internal master by
`tools/prepare_public_master.py`. Rows, columns and every classification field are
identical. Two kinds of text were changed:

- Antwerp geoportal and Bogotá SIMUR descriptions name the staff member who maintains
  each layer. Those names and e-mail addresses were replaced by `[redacted]` or
  `[email redacted]` (114 substitutions in 51 descriptions).
- The review notes in `round_tag` were edited for publication: neutral wording ("after
  review"), words run together by line wrapping separated again, dashes normalised, and
  the workbook column letter replaced by the field name (937 notes).

## Framework

`framework_v5_1.csv` lists the 51 indicators of framework v5.1 with their domain,
cluster, operationalisation and the paper's reference numbers for each.

## Licence

CC0 1.0 (see `LICENSE`). Titles and descriptions are catalogue metadata published by
the cities and agencies listed in `source`; the links lead to the data under the terms
each publisher sets.
