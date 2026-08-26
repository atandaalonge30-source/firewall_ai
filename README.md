# IntelliFirewall — Intelligent Firewall Rule Recommendation System

A Flask + scikit-learn web application that analyzes network traffic features,
classifies traffic as **normal** or **malicious** using a trained Random Forest
model, and automatically generates a recommended firewall rule (ALLOW / DENY)
for each record.

## Why Python/Flask, and not XAMPP or Vercel?

This project is a **machine learning system**, and the model is trained and
served with Python (scikit-learn). XAMPP is a PHP + MySQL stack — it cannot
load or run a scikit-learn model directly. Vercel is built for static sites
and short-lived serverless JavaScript functions, not a persistent Python
service holding a loaded ML model in memory. **Render** runs a normal
long-lived Python web process, which is exactly what this app needs, so it
is the recommended host below. If a PHP/MySQL front-end is required for a
separate part of your coursework, this Flask app can still be called from
PHP as a remote API, but it cannot itself run inside XAMPP.

## Project Structure

```
firewall_ai/
├── app.py                  # Flask application & routes
├── requirements.txt
├── Procfile                 # process command for Render
├── render.yaml               # Render infrastructure-as-code
├── data/
│   ├── generate_data.py     # generates the training dataset
│   └── traffic_data.csv     # generated dataset (replace with real public dataset if desired)
├── model/
│   ├── train_model.py       # trains and saves the ML model
│   ├── model.pkl, scaler.pkl, encoders.pkl, label_encoder.pkl, feature_cols.pkl
│   └── metrics.json         # accuracy / precision / recall / F1 of the trained model
├── utils/
│   ├── features.py          # preprocessing + Predictor class
│   └── rules.py              # rule recommendation engine
├── templates/                # HTML pages (Dashboard, Analyze, Upload, Results, Rules)
└── static/css/style.css
```

## 1. Run Locally

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset and train the model (only needed once,
#    or again if you replace data/traffic_data.csv with a real dataset)
python data/generate_data.py
python model/train_model.py

# 4. Run the app
python app.py
```

Then open **http://localhost:5000** in your browser.

### Using the real public dataset instead of the generated sample

The project ships with a generated sample dataset (`data/traffic_data.csv`)
so it runs immediately without an internet download. To use a real dataset
(e.g. NSL-KDD or CICIDS2017 from Kaggle) as described in the write-up's
Methodology:

1. Download the dataset CSV from Kaggle.
2. Rename/re-map its columns to match: `duration, protocol_type, service,
   src_bytes, dst_bytes, count, srv_count, serror_rate, same_srv_rate,
   dst_host_count, label` (label values: `normal` / `malicious`).
3. Replace `data/traffic_data.csv` with this file.
4. Re-run `python model/train_model.py`.

No other code changes are required.

## 2. Push the Code to GitHub

```bash
cd firewall_ai
git init
git add .
git commit -m "Initial commit - IntelliFirewall"
git branch -M main
git remote add origin https://github.com/<your-username>/intelli-firewall.git
git push -u origin main
```

## 3. Deploy to Render (recommended)

1. Go to https://render.com and sign in (you can sign in with GitHub).
2. Click **New +** → **Web Service**.
3. Connect your GitHub repository (`intelli-firewall`).
4. Render will detect `render.yaml` automatically. If asked manually, set:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt && python model/train_model.py`
   - **Start Command:** `gunicorn app:app`
5. Click **Create Web Service**. Render will build and deploy the app and
   give you a live URL such as `https://intelli-firewall.onrender.com`.
6. Every time you `git push` to `main`, Render automatically redeploys.

That's it — no server management needed.

## 4. Notes on Vercel

If you are required to also show a Vercel deployment, Vercel can only host
the **static front-end** (HTML/CSS) — it cannot run the Flask/ML backend as
a persistent service on the free tier. If this is required, the practical
approach is to deploy the Flask API on Render (as above) and, separately,
deploy a lightweight static page on Vercel that calls the Render API over
HTTPS using `fetch()`. This is optional and not necessary for the project
to work — Render alone fully hosts the working system.

## 5. Notes on XAMPP

XAMPP is included here only for completeness of understanding: it can run
the project's **static exported reports** (e.g., if you export
`data/traffic_data.csv` or `model/metrics.json` and build a simple PHP
dashboard reading from a MySQL table) but it cannot execute the Python
model itself. For the machine learning classification to work, the Flask
app must run on a Python-capable host such as Render, your own PC, or a
Python-enabled server — not XAMPP.
