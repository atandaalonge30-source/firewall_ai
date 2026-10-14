# Fake News Detection System Using Natural Language Processing

This is an HND project implementation matching the supplied proposal: it treats fake-news detection as a **binary text-classification** problem. It trains a TF-IDF (unigram + bigram) and Logistic Regression model, uses train/validation/test splits, saves the trained pipeline, creates evaluation reports, and provides a Flask web interface for predictions and workflow results.

> Important: a model prediction is not proof that a story is false or true. It detects patterns learned from its training data. For real use, combine it with source checks, dates, primary evidence and professional fact-checking.

## 1. Prerequisites

Install Python 3.10 or newer. Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass`, then activate again. On macOS/Linux use `source .venv/bin/activate`.

## 2. Obtain and prepare data

Use a legally accessible labelled fake/real news dataset. A common student option is the Kaggle **Fake and Real News Dataset**, which has `Fake.csv` and `True.csv` files with `title` and `text` columns. Download it yourself, then place the files in `data/`:

```
data/Fake.csv
data/True.csv
```

Do **not** evaluate only on the four-row `sample_news.csv`; it exists solely to show the expected format. Your data should be reasonably balanced and contain enough records for all splits. Remove blank, duplicate and irrelevant records; keep the source and collection date in a separate data log for your project report.

If your dataset is one CSV instead, it must contain `label` plus `text` (or `title` + `text`) columns. `label` may be `fake`/`real` or `0`/`1` (`0 = fake`, `1 = real`). A CSV with a `target` column is also supported for the linked Fake and Real News Dataset (`0 = real`, `1 = fake`).

For this project, the trained demonstration uses the educational [Fake and Real News Dataset mirror on Hugging Face](https://huggingface.co/datasets/haoxianc/gilchr_fake-and-real-news-dataset), accessed 2026-10-13. The dataset card credits Gilchrist, declares MIT, and links to the [Kaggle source dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). Cite the dataset card and original source in your report, preserve the MIT notice when redistributing covered material, and verify the original source terms for your own use. The benchmark has inherited publisher, topic, period, and label biases; results do not establish real-world fact-checking accuracy.

To train with that dataset after downloading `fake_real_news.csv` into `data/`, run:

```powershell
python -m src.train --data data/fake_real_news.csv --additional-real-data data/additional_real_examples.csv --additional-fake-data data/additional_fake_examples.csv
```

The additional data files contain 12 user-labelled REAL examples and 13
user-labelled FAKE examples. They are appended only to the training partition,
not the validation or test data. Their labels have not been independently
fact-checked, so these examples are a small supervised-training addition—not
evidence of better overall accuracy or a claim-verification feature.
Each supplemental example is weighted 50x during fitting so its supplied
label has a meaningful effect despite the much larger base dataset.

## 3. Train and evaluate

From the project root, for the two-file dataset run:

```powershell
python -m src.train --fake-csv data/Fake.csv --true-csv data/True.csv
```

Or, for a single labelled CSV:

```powershell
python -m src.train --data data/my_labelled_news.csv
```

The script does the following reproducibly (random seed 42):

1. Combines title and article body, removes blank/very short items and standardizes labels.
2. Normalizes text and removes duplicate normalized articles **before** splitting. If identical normalized text has conflicting labels, all copies are excluded rather than arbitrarily assigning one label.
3. Stratifies the remaining data into 70% training, 15% validation and 15% held-out test data.
4. Extracts TF-IDF unigram and bigram features, excluding English stop words.
5. Fits class-balanced Logistic Regression on the training portion.
6. Reports validation and held-out test accuracy, precision, recall, F1 and error counts; saves the fitted pipeline, confusion matrix, and feature-weight summary.

Outputs go to `artifacts/`:

- `fake_news_model.joblib` — trained model for reuse.
- `metrics.json` — numeric results for Chapter Four.
- `test_predictions.csv` — actual vs predicted records for error analysis.
- `confusion_matrix.png` — visual evaluation result.

The metrics report also includes a cleaned-data profile, the duplicate/conflicting-label
removal counts, dataset attribution, and highest-weighted model terms for the workflow
report. `test_predictions.csv` remains local/ignored because it contains article text;
use it to inspect errors in your project analysis without republishing the records.

## 4. Test a single item from the terminal

```powershell
python -m src.predict --text "Paste a news headline or full article here."
```

## 5. Run the web system

After training:

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in a browser. Paste text and select **Analyse text**. Stop the server with `Ctrl+C`.

Open `/workflow` to view the data profile and class distribution, preprocessing steps,
stratified split sizes, model evaluation metrics, confusion matrix, and top weighted
TF-IDF terms by predicted class. Feature terms are learned associations, not causal
evidence. The workflow report is generated during training and is served from
`artifacts/metrics.json`; the confusion matrix is served from
`artifacts/confusion_matrix.png`.

## 6. Publish the web app

Render is the simplest option for this Flask application. Vercel is primarily
designed for frontend and serverless applications; this app needs a persistent
Python web process and a trained scikit-learn model, so Render is the more
straightforward fit.

1. Download a labelled dataset and train the model using one of the commands in
   section 3. Confirm that `artifacts/fake_news_model.joblib` was created.
   Include `artifacts/metrics.json` and `artifacts/confusion_matrix.png` as well
   so the deployed workflow report has its training summary and chart.
2. Put this project folder in a GitHub repository. The trained model is
   git-ignored along with generated reports, so it must be explicitly added:

   ```powershell
   git add -f artifacts/fake_news_model.joblib
   git add app.py requirements.txt README.md src
   git commit -m "Prepare app for deployment"
   git push
   ```

   Only deploy a model trained on data you are allowed to redistribute. If the
   model is too large for GitHub, store it in an external model store and set
   Render's `MODEL_PATH` environment variable to a downloaded model path.
3. To use the existing `firewall_ai` Render service, open **Settings → Build &
   Deploy** and set its **Root Directory** to `news_classifier`. This keeps
   the existing service and the old firewall project files in the repository,
   while serving this classifier from the subfolder. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn --bind 0.0.0.0:$PORT app:app`

   If creating a separate service instead, connect the GitHub repository,
   choose Python, and use the same build and start commands:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn --bind 0.0.0.0:$PORT app:app`
4. Deploy. Render gives the service a public `*.onrender.com` URL. Open that
   URL to use the app; `/health` should return `{"status":"ok"}` when the model
   is present.

The trained model is included. To retrain it, provide a labelled dataset as
described above; the included sample CSV is only a format example and is too
small for training. Do not treat predictions or their confidence as proof that
a story is true or false.

## Academic evaluation and responsible presentation

See [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for the current data citation,
deduplication method, measured results, confusion matrix counts, feature
discussion, limitations, and recommendations.

The `/workflow` page and `artifacts/metrics.json` report the actual cleaned sample
counts, class balance, duplicates removed, split sizes, validation/test metrics,
confusion-matrix error totals and weighted terms. Explain these metrics in your own
words in the report; include both strengths and limitations.

Exact normalized-article duplicates are excluded before the split, which prevents
the same normalized record from appearing in multiple partitions. This alone does
not prove independent generalization: records from the same publisher, political
topic or period may still appear in both training and test sets. A stronger follow-up
study should use publisher/source-grouped or time-based holdout data and an
independent dataset, selected before evaluating the final model.

Use cautious wording such as **“classifies text using patterns learned from this
dataset”**, not “verifies news.” The displayed feature terms are statistical
associations, not causal reasons or fact evidence. Discuss false positives and
false negatives from the held-out `test_predictions.csv`, dataset provenance,
licence, label quality, class imbalance, and the possibility that modern or
non-political stories differ from this benchmark.

## 7. Run basic automated tests

```powershell
pytest -q
```

## Suggested Chapter Four report structure

1. Cite Gilchrist's [dataset card](https://huggingface.co/datasets/haoxianc/gilchr_fake-and-real-news-dataset) and its linked Kaggle source; report its declared licence and access date.
2. State the rows before and after normalized deduplication, number removed, final class counts and median text length.
3. Explain the 70/15/15 stratified split, random seed 42, and why the held-out test set was not used to choose the model.
4. Describe text normalization, TF-IDF unigrams/bigrams and class-balanced Logistic Regression.
5. Insert `artifacts/confusion_matrix.png` and report accuracy, per-class precision, recall and F1 from `artifacts/metrics.json`.
6. Discuss false positives and false negatives using local `artifacts/test_predictions.csv`; omit or anonymize article text in public material.
7. Interpret feature weights cautiously and state the source, label, publisher and temporal limitations; recommend independent and group/time-based evaluation.

## Project layout

```
app.py                    Flask user interface
src/text_processing.py    NLP normalization
src/train.py              data loading, model training and evaluation
src/predict.py            command-line prediction
tests/                    basic unit tests
data/                     place downloaded data here
artifacts/                created model and evaluation outputs
```
