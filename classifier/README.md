# Classifier (Appendix A)

A support vector machine that assigns a catalogue record's English description to
one of seven GAPPA-derived domains or to *Other*. It was used to screen 65,022
catalogued records (prediction run of 11 February 2026) before every candidate
was reviewed by hand. It assisted screening; it did not decide inclusion.

| | |
|---|---|
| Embeddings | `sentence-transformers/all-mpnet-base-v2`, 768 dimensions, not normalised, full text |
| Model | `SVC(kernel="rbf", C=100, gamma="scale", class_weight="balanced", probability=True)` |
| Selection | grid search over 96 configurations (kernel, C, gamma, class weight), 5-fold CV |
| Classes | 8: Active Transport Environment, Air Quality, NCD's, Other, Recreational PA, Recreational and Sports Environment, Total PA, Transport PA |
| Training set | 5,886 texts: seed set, documents reviewed during active learning, and their synthetic paraphrases |
| Static holdout | n = 130, reserved from the seed set before active learning |
| Dynamic holdout | n = 152 in the paper's evaluation, split by document group |

## Quick start

```bash
pip install -r classifier/requirements.txt
python classifier/predict_example.py "Number of cyclists counted per street segment"
python classifier/predict_example.py --file catalogue.csv --column description --out labelled.csv
```

The first run downloads the embedding model (about 420 MB) from Hugging Face.
The script returns the top two labels with their probabilities and flags
predictions below 0.70, the final review threshold of the active-learning loop.

## Contents

```
data/
  classifier.pkl                      the published model (scikit-learn 1.7.2, 2,884 support vectors)
  classifier_config.txt               configuration written at model selection, 31 January 2026
  label_maps.json                     class index -> label -> framework domain
  docs_and_labels_augmented.xlsx      final training set (docs, labels)
  sampled_docs_df_04022026.xlsx       seed set of 648: 600 catalogue records (100 per city, seed 92) + 48 indicators
  new_base_file_docs_and_labels.xlsx  the 48 candidate indicators from the literature review
  holdout_static.xlsx                 static holdout
  holdout_dynamic.xlsx                dynamic holdout pool
scripts/
  citymove_classification_v8.py       active-learning pipeline (train, predict, review, augment)
  synthetic_text_generation.py        paraphrase generation through a local llama.cpp server
  citymove_diagnostics.py             confidence, UMAP, c-TF-IDF and holdout diagnostics
  sampling_script_training_docs.py    draws the seed sample (100 per catalogue export, seed 92)
  model_selection/                    classifier comparison, SVM grid search, save best model
  archive/                            the 31 January 2026 augmentation script
outputs/
  model_selection/                    comparison and grid-search results
  validation/                         holdout reports and confusion matrices (6 February 2026)
  diagnostics/                        diagnostic tables and figures
predict_example.py                    apply the model to new descriptions
PROVENANCE.md                         how the published files relate to the paper
```

The scripts read their files from `classifier/data`; set `CITYMOVE_CLASSIFIER_DATA`
to point them elsewhere. Rerunning the full pipeline also needs the catalogue
exports (`to_predict_df.xlsx`, 65,464 rows), which will be deposited with the
analysis files.

Descriptions harvested from Antwerp's geoportal named the staff member who
maintains each layer. Those names and e-mail addresses are redacted in the
workbooks here (72 substitutions in 41 texts, rules in `../tools/pii.py`).
See [PROVENANCE.md](PROVENANCE.md) for what that means for exact
reproduction, and for the points where the scripts differ from the paper's
description.
