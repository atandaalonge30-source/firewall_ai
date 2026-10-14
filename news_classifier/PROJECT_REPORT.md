# Fake News Text Classification: Project Report

> This report summarizes the implemented educational prototype and its measured
> results. Adapt the presentation to your institution's format, add your own
> project context, and follow your supervisor's citation requirements. This
> system classifies text patterns; it does not verify claims.

## Abstract

This project implements a web-based binary text classifier that predicts
whether an input news article resembles the `fake` or `real` examples in its
training dataset. It combines article titles and content, normalizes text,
removes duplicate normalized records before data splitting, converts text to
TF-IDF unigram and bigram features, and trains a class-weight-balanced
Logistic Regression classifier. On a stratified held-out test set of 5,824
deduplicated records, the current model achieved 98.88% accuracy. This is a
benchmark result on one dataset, not evidence of accuracy on current news or a
fact-checking capability. Dataset/source bias and the random split limit the
conclusions that can be drawn.

## Aim and scope

The aim is to demonstrate an end-to-end natural-language-processing workflow:
labelled data input, exploratory summary, text preprocessing, stratified
training/validation/test splitting, model fitting, evaluation, error
reporting, feature-weight inspection, and a Flask interface for predictions
and results. The prototype does not retrieve sources, corroborate claims, or
decide whether a real-world event happened.

## Dataset and attribution

The trained demonstration uses the educational **Fake and Real News Dataset**
mirror published by Gilchrist on Hugging Face. Its dataset card declares an MIT
license and links to the underlying Kaggle data. The card describes `title`,
`content`, and `target` fields; the model combines title/content and maps
`target` 0 to `real` and 1 to `fake`.

- Dataset card and mirror: <https://huggingface.co/datasets/haoxianc/gilchr_fake-and-real-news-dataset>
- Kaggle dataset credited by the mirror: <https://www.kaggle.com/datasets/gilchr/fake-and-real-news-dataset>
- Related original merged source named in the dataset description: <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>
- Access date for the mirror: 2026-10-13
- License: MIT, as declared by the mirror's dataset card. Preserve the notice
  when redistributing covered material and check the upstream terms.

After label normalization and the 40-character minimum, 44,895 records
remained. Texts were normalized for duplicate detection; 6,071 repeated
records were removed and no normalized-text groups had conflicting labels.
The final modelling data contains 38,824 unique normalized texts:

| Label | Records | Share |
|---|---:|---:|
| Fake | 17,899 | 46.1% |
| Real | 20,925 | 53.9% |

Median combined article length is 2,288 characters. These counts describe this
cleaning pipeline and dataset copy; they should not be presented as the
original publisher's raw dataset counts.

## Method

1. Combine each available title and article body.
2. Remove missing labels/text and records shorter than 40 characters; map the
   supplied labels to `fake` and `real`.
3. Lowercase and normalize URLs, HTML, punctuation, and whitespace. Group on
   this normalized text before splitting. Repeated texts with the same label
   are reduced to one record. If a normalized text has conflicting labels,
   all its copies are excluded.
4. Split the remaining records with stratification and random seed 42:
   70% training (27,176), 15% validation (5,824), and 15% test (5,824).
   The validation and test partitions are separate; the held-out test set was
   not used to fit the model.
5. Apply scikit-learn TF-IDF with unigrams and bigrams, English stop words,
   `min_df=2`, `max_df=0.95`, sublinear term frequency, and up to 100,000
   features.
6. Fit Logistic Regression (`max_iter=2000`, `class_weight="balanced"`,
   random seed 42). No records are synthetically oversampled.
7. Report accuracy, per-class precision/recall/F1, a confusion matrix, and
   test-set error counts. Inspect the largest signed linear-model weights as
   feature associations.
8. Add 12 user-labelled REAL examples and 13 user-labelled FAKE examples from
   `data/additional_real_examples.csv` and `data/additional_fake_examples.csv`
   to the training partition only. They are not independently fact-checked
   and are excluded from validation and held-out test metrics; each has a
   training weight of 50.

## Evaluation results

| Held-out test metric | Fake | Real |
|---|---:|---:|
| Precision | 99.03% | 98.76% |
| Recall | 98.55% | 99.17% |
| F1-score | 98.79% | 98.97% |

Overall test accuracy is **98.88%** (macro F1: **98.88%**). Validation
accuracy is **98.88%**. The held-out test set contains 5,824 examples: 2,685
fake and 3,139 real.

Confusion matrix (rows are actual labels, columns are predicted labels; order
is `fake`, then `real`):

| Actual \ Predicted | Fake | Real |
|---|---:|---:|
| Fake | 2,646 | 39 |
| Real | 26 | 3,113 |

There were 65 incorrect test predictions: 39 fake-labelled articles were
predicted real, and 26 real-labelled articles were predicted fake. See the
generated `artifacts/confusion_matrix.png` and the local, git-ignored
`artifacts/test_predictions.csv` for the chart and record-level error analysis.
Do not publish article text from the error file without checking its source
terms and privacy/copyright implications.

## Feature analysis

The largest weights associated with the fake class include terms such as
`video`, `image`, `featured`, and political names/phrases. Terms associated
with the real class include `reuters`, `said`, and `washington reuters`.
These weights show correlations in the benchmark—not causal explanations or
evidence that any story is true or false. In particular, publisher names among
the strongest real-associated terms are a warning that the classifier may
learn source/style shortcuts.

## Discussion and limitations

The evaluation shows that the selected model separates examples from this
dataset's held-out random partition well after exact normalized duplicates are
removed. It does **not** establish equivalent performance on new, current, or
independently sourced articles. A random stratified split can still put the
same publisher, subject, event, or time-period patterns in both training and
test sets. The feature weights' association with publisher/style terms
illustrates this risk.

Other limitations include inherited labels and dataset curation decisions,
the dataset's time/topic/source coverage, possible class/source correlations,
and distribution shift as language and events change. The displayed
"confidence" is a model probability, not a calibrated probability that a
claim is false. Predictions should not be used for moderation, reputational
judgments, or consequential decisions without independent evidence and
validation.

## Recommended further work

- Evaluate with a publisher/source-grouped split and a time-based holdout, and
  compare both to the current random-split baseline.
- Test on a separate, documented dataset collected under appropriate terms;
  inspect labels and record the collection date.
- Review false positives and false negatives, investigate source shortcuts,
  and report uncertainty/calibration if confidence is displayed.
- Expand EDA with source, subject, date, missing-value, and duplicate
  breakdowns where those fields are available and their use is appropriate.
- Keep the public description explicit that this is a screening/classification
  demonstration rather than an automated fact-checker.

## Reproducibility and outputs

From the project root, with requirements installed:

```powershell
python -m src.train --data data/fake_real_news.csv --additional-real-data data/additional_real_examples.csv --additional-fake-data data/additional_fake_examples.csv
pytest -q
```

Training writes `artifacts/fake_news_model.joblib`,
`artifacts/metrics.json`, `artifacts/confusion_matrix.png`, and
`artifacts/test_predictions.csv`. The workflow page at `/workflow` displays
the report summary, evaluation, confusion matrix, and feature associations.
