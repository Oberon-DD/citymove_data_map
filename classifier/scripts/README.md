# Training the classifier

The model in [`../data/classifier.pkl`](../data) was trained with the two scripts in
this folder. [`citymove_classification_v8.py`](citymove_classification_v8.py) runs the
active-learning loop; [`synthetic_text_generation.py`](synthetic_text_generation.py)
writes the paraphrases that augment every reviewed document.

## How the loop works

1. **Start.** If the data folder holds a training set (`docs_and_labels_augmented.xlsx`),
   the script loads it with the saved model. Otherwise it starts from the seed set
   (`sampled_docs_df_04022026.xlsx`, 648 texts) and keeps 20% aside as the static
   holdout (stratified, seed 42).
2. **Batches.** The records to screen (`to_predict_df.xlsx`, one description per row in a
   column named `doc`) are shuffled into batches of 1,000.
3. **Review.** The model labels every record in the batch. The (at most) 20 least
   confident predictions below 0.70 are shown one at a time in the console:
   `Y` accepts the label, `N` offers the second-best label, a second `N` asks you to type
   the correct one, and `EXIT` saves and stops.
4. **Augment.** Each reviewed document enters the training set together with paraphrases
   from a local language model: 20 for classes under 100 documents, 15 for 100 to 250,
   10 above 250, and `OTHER_AUGMENTATIONS` for *Other*. A document and its paraphrases
   share a `doc_group_id`, so they never end up on both sides of an evaluation split.
5. **Retrain and evaluate.** The SVM is refitted after each reviewed batch. Every third
   batch, a temporary SVM trained on 80% of the document groups is scored on the
   originals of the remaining 20%: the dynamic holdout.
6. **Save.** After every batch the model, training set and holdout are written to the
   data folder, with a timestamped copy in `backups/`. Once all batches are done, the
   script scores both holdouts and writes predictions for the whole target file to
   `all_predictions.xlsx`.

The confidence threshold rose from 0.50 to 0.60 and then 0.70 during the project;
`CONFIDENCE_THRESHOLD` holds the final value. Other settings sit at the top of the script.

## Running the trainer

The trainer overwrites the model and the training set in its data folder, so work on a
copy and point the script at it:

```bash
pip install -r classifier/requirements.txt
cp -R classifier/data ~/citymove_training
export CITYMOVE_CLASSIFIER_DATA=~/citymove_training     # on Windows: set CITYMOVE_CLASSIFIER_DATA=...
python classifier/scripts/citymove_classification_v8.py
```

Put the records you want to screen in `to_predict_df.xlsx` in that folder, one description
per row in a column named `doc`, in English. With the published training set in the folder,
the loop continues from the published model; remove `docs_and_labels_augmented.xlsx` and
`classifier.pkl` from the copy to start again from the seed set. The catalogue file used in
the paper (65,464 records) will be deposited with the analysis files.

## Running the paraphraser

`synthetic_text_generation.py` sends each request to a llama.cpp server at
`http://localhost:8080` (endpoint `/completion`). For the paper this served
Phi-3-Mini-4K-Instruct, quantised to 8 bits, through Anaconda AI Navigator; any
llama.cpp server works, for example:

```bash
llama-server -m path/to/Phi-3-mini-4k-instruct-Q8_0.gguf --port 8080
python classifier/scripts/synthetic_text_generation.py    # three variants for each of three test descriptions
```

Each request asks for one paraphrase ("Create ONE concise variant (1-2 sentences) of this
dataset description. Rephrase using different words but keep the same meaning.") with
temperature 0.8, top-k 50, top-p 0.92 and at most 80 new tokens. The prompt never sees the
label. Replies are cleaned (list markers and preambles removed, first line kept, 20 to 500
characters) and exact duplicates dropped, with up to four attempts per requested variant.
Set `USE_LLM = False` in the trainer to train without paraphrases.

## Other scripts

| Script | Purpose |
|---|---|
| [`citymove_diagnostics.py`](citymove_diagnostics.py) | Confidence by city and label, UMAP projection, c-TF-IDF terms per label, holdout reports and threshold sweeps (`../outputs/diagnostics`, `../outputs/validation`). Needs the catalogue file as well. |
| [`sampling_script_training_docs.py`](sampling_script_training_docs.py) | Draws the seed sample: 100 records per catalogue export, seed 92. Needs the catalogue exports (`CITYMOVE_CITY_DATA`). |
| [`model_selection/`](model_selection) | Classifier and embedding comparison, the SVM grid search (96 configurations, 5-fold CV) and saving the chosen model (`../outputs/model_selection`). |
| [`archive/`](archive) | The 31 January 2026 version of the paraphraser. |

## Reproducibility

- `STRIP_TITLES = False` embeds full texts, as for the published model. `True` restores the
  exploratory setting that embeds only the part of a text before its first colon.
- With `STRIP_TITLES = False` the trainer's own evaluation of the published model on the
  static holdout gives weighted F1 0.947 and macro F1 0.858.
- Batches are shuffled without a fixed seed and the published model has no fixed random
  state, so a new run will not rebuild `classifier.pkl` bit for bit.
- The published training set has no `is_synthetic` or `doc_group_id` columns. Continuing
  from it, every text counts as an original in its own group, so dynamic-holdout scores
  will be optimistic until new groups are added.
- [`../PROVENANCE.md`](../PROVENANCE.md) lists where the scripts differ from the paper.
