# Provenance of the published classifier

This note records how the files in this folder relate to Appendix A of the paper,
what was checked, and where the scripts as saved differ from the paper's
description. It is written so the model can be audited, not only reused.

## Timeline

| Date (2026) | Step | Files |
|---|---|---|
| 31 January | Model selection: classifier and embedding comparison, SVM grid search | `outputs/model_selection/`, `data/classifier_config.txt`, `scripts/model_selection/` |
| 4 February | Seed set fixed (600 catalogue records + 48 indicators) | `data/sampled_docs_df_04022026.xlsx` |
| 6 February | Holdout evaluation reported in the paper | `outputs/validation/classification_report_*.txt` |
| 11 February | Final model and training set; prediction run over 65,022 catalogued records | `data/classifier.pkl`, `data/docs_and_labels_augmented.xlsx` |
| April | Scripts consolidated for the paper | `scripts/` |

## Checked facts

- `classifier.pkl` is `SVC(kernel="rbf", C=100, gamma="scale", class_weight="balanced", probability=True)`
  with 768 input features, 8 classes and 2,884 support vectors, pickled with scikit-learn 1.7.2.
- The model was trained on **full texts**. Re-embedding 300 training documents in full
  reproduces a support vector exactly for 214 of them (median distance 0.000); embedding
  the same documents after the title stripping in `citymove_classification_v8.py`
  reproduces none (median distance 0.644).
- Class indices follow the alphabetical label order (`data/label_maps.json`); the
  support-vector counts per class match the class sizes of the training set.
- Static holdout (n = 130), scored with the published model and the pipeline's rule
  (label = highest probability): accuracy 0.946, weighted F1 0.947, macro F1 0.858. The
  paper reports 0.96, 0.96 and 0.89 from the 6 February evaluation, five days before the
  final model. Two of the 130 holdout texts also occur in the final training set. The
  holdout is dominated by *Other* (109 of 130), so accuracy flatters the smaller classes.
- Dynamic holdout: the paper's figures (n = 152, accuracy 0.74, weighted and macro
  F1 0.74) come from a temporary SVM trained on 80% of the document groups. The group
  identifiers were not saved with the final training set, so this evaluation cannot be
  recomputed from the files here. `data/holdout_dynamic.xlsx` holds the 212-row pool.

## Where the scripts differ from the paper

1. **Title stripping.** `EmbeddingCache.encode` in `citymove_classification_v8.py` passes
   each text through `extract_discriminative_text`, which keeps only the part before the
   first colon when that part has at least 15 characters. The published model was trained
   without this step (see above), as the paper describes. Disable it to reproduce the model.
2. **Augmentation model.** The paper names Phi-3-Mini-4K-Instruct (8-bit, llama.cpp via
   Anaconda AI Navigator). The docstring of `synthetic_text_generation.py` names
   Mistral-7B-Instruct-v0.1; the code itself calls whatever model the local llama.cpp
   server at `http://localhost:8080` serves.
3. **Variants for Other.** The paper gives 10 paraphrases per *Other* document;
   `OTHER_AUGMENTATIONS` in `citymove_classification_v8.py` is 2.
4. **c-TF-IDF.** The paper computes the terms per label "via supervised BERTopic";
   `citymove_diagnostics.py` computes the same class-based TF-IDF directly, without fitting
   a BERTopic model.
5. **Random state.** `classifier_config.txt` (31 January) lists `random_state=42`; the
   published model carries `random_state=None`. Refitting therefore does not reproduce the
   Platt-scaled probabilities bit for bit.

## Changes made for publication

- Maintainer names and e-mail addresses embedded in Antwerp geoportal descriptions are
  redacted in `docs_and_labels_augmented.xlsx` (39 texts) and `holdout_dynamic.xlsx`
  (2 texts), 72 substitutions in total, by `tools/redact_classifier_data.py`. The model
  is unchanged. These 41 texts no longer embed to the vectors the model saw, so a refit
  on the published training set is close to, but not identical with, `classifier.pkl`.
- Local folder paths in the scripts and in `classifier_config.txt` were replaced; the
  scripts now read `classifier/data` or the folder named in `CITYMOVE_CLASSIFIER_DATA`.
  No other line of the scripts was changed.

## Held back for the analysis deposit

The catalogue-level input (`to_predict_df.xlsx`, 65,464 records from the six cities'
catalogues) and the prediction output (`target_predictions_with_confidence.xlsx`, 65,022
rows) will be deposited with the analysis files, together with the catalogue exports
they were built from.
