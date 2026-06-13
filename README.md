<div align="center">

# 🧠 RealTime ML

Machine Learning predictions served live through the browser

Trained ensemble models taken out of Jupyter notebooks and deployed behind a Django web app — fill in a form, get an instant prediction. No notebooks, no code, just results in real time

<p>
  <img src="https://img.shields.io/badge/Python-3.11.5-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.11.5" />
  <img src="https://img.shields.io/badge/Django-5.0.2-092E20?style=flat&logo=django&logoColor=white" alt="Django 5.0.2" />
  <img src="https://img.shields.io/badge/scikit--learn-1.2.1-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn 1.2.1" />
  <img src="https://img.shields.io/badge/pandas-2.2.0-150458?style=flat&logo=pandas&logoColor=white" alt="pandas 2.2.0" />
  <img src="https://img.shields.io/badge/NumPy-1.26.4-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy 1.26.4" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter Notebook" />
</p>
</div>
<br/>

## ✨ Overview

**RealTime ML** demonstrates the *full lifecycle* of a practical machine learning project — from raw data to a clickable prediction in the browser:

> **Data exploration** → **feature engineering** → **model benchmarking** → **ensemble selection** → **serialization** → **live web deployment**

It ships with **two independent prediction services** built on two different ML problem types, both powered by **stacking ensembles** chosen after benchmarking multiple algorithms against simpler baselines.

| 🩺 Service | 🎯 Problem Type | ❓ Question It Answers |
|---|---|---|
| **Heart Disease Detection** | Classification | Given a patient's clinical attributes, is heart disease likely present? |
| **Fuel Consumption Prediction** | Regression | Given a car's specifications, what is its estimated fuel economy (MPG)? |
<br/>

## 🚀 Highlights

- 🧩 **Ensemble modeling** — Voting *and* Stacking ensembles benchmarked against tuned individual models (Logistic Regression, SVM, k-NN, Decision Trees, Random Forest, and more).
- ⚙️ **Production-style inference** — preprocessing (Min–Max scaling + one-hot encoding) is reproduced *exactly* at request time so live predictions match training.
- 🔍 **Systematic tuning** — every candidate model optimized with `GridSearchCV` and compared on consistent metrics (precision/accuracy vs. R²).
- 💾 **Decoupled training & serving** — models and scalers serialized with `pickle` and loaded by the web app, so serving needs no retraining.
<br/>

## 🛠️ Tech Stack

**Web**
- Python 3.11.5
- Django 5.0.2
- SQLite

**Machine Learning & Data**
- scikit-learn 1.2.1
- pandas 2.2.0
- NumPy 1.26.4
- Pillow 10.2.0
- pickle for serialization

**Notebooks & Visualization**
- Jupyter Notebook
- Matplotlib
- Seaborn

**Datasets**
- `heart.csv` — clinical heart-disease data (classification)
- `auto-mpg.csv` — automobile fuel-economy data (regression)
<br/>

## 🎛️ Features & Functionality

- **Two live ML services** behind a single web app — classification and regression.
- **Interactive web forms** collecting the exact features each model expects.
- **Real-time inference** — user input is converted to a DataFrame, preprocessed, and scored on every request.
- **Automatic preprocessing pipeline:**
  - Min–Max scaling of continuous heart-disease features via a pre-fitted scaler (`min_max_scaler.pkl`).
  - One-hot encoding of categorical inputs (sex, chest-pain type, ECG, exercise angina, ST slope) reconstructed to match the training schema.
- **Pre-trained, serialized models** loaded from `.pkl` files — no retraining to serve predictions.
- **Human-readable results** on a dedicated page (e.g. *"Heart disease detected!"* or *"The estimated gas consumption is: 27.431 MPG"*).
<br/>

## ⚡ Quick Start

### 1. Clone the repository
```bash
# 1. Clone the repository
git clone https://github.com/HarshTanwar1/RealTime_ML.git
cd RealTime_ML
```

### 2. (Recommended) Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install numpy==1.26.4 pandas==2.2.0 pillow==10.2.0 django==5.0.2 scikit-learn==1.2.1
```

### 4. Apply database migrations
```bash
python manage.py migrate
```

### 5. Start the development server
```bash
python manage.py runserver
```

Then open **http://127.0.0.1:8000/** in your browser, pick a service, fill in the form, and get a live prediction. 🎉

> ⚠️ **Version note:** the models were pickled with **scikit-learn 1.2.1**. A different version may trigger unpickling warnings or errors — matching the version is recommended
<br/>

## 🗂️ Project Structure

```
RealTime_ML-main/
├── manage.py                       # Django entry point
├── RealTime_ML/                    # Django project config (settings, urls, wsgi/asgi)
├── ML/                             # Main Django app (views & urls — the inference logic)
├── Templates/                      # HTML templates (index, classification, regression, prediction)
├── Classification/                 # Heart disease notebook + heart.csv
├── Regression/                     # MPG prediction notebook + auto-mpg.csv
├── stacking_classifier_model.pkl   # Trained heart-disease ensemble
├── stacking_regressor_model.pkl    # Trained MPG ensemble
├── min_max_scaler.pkl              # Fitted scaler for classification features
└── db.sqlite3                      # Django database
```
<br/>

## 🧪 The Process Behind It

- **Data exploration** — loaded and inspected both datasets in Jupyter, handled missing values, and studied feature distributions with Matplotlib/Seaborn.
- **Feature engineering** — one-hot encoded categoricals (`pd.get_dummies`) and Min–Max scaled continuous features for the heart-disease data; cleaned and prepared the auto-mpg data.
- **Model benchmarking** — trained and tuned a range of algorithms with `GridSearchCV`:
   - *Classification:* Gaussian Naive Bayes, Logistic Regression, SVM, k-NN, Decision Tree.
   - *Regression:* Random Forest, Linear Regression, Decision Tree, ElasticNet, Lasso, k-NN.
- **Ensembling** — compared **Voting** and **Stacking** ensembles built from the best models, using `GridSearchCV` to select the optimal `final_estimator`.
- **Model selection & serialization** — picked the best stacking ensemble for each task and saved the models (and scaler) with `pickle`.
- **Web integration** — built a Django app that loads the models, reproduces the exact preprocessing at request time, runs inference, and renders results.
<br/>

## 📚 What I Learned

- **Bridging the notebook-to-production gap** — preprocessing must be *replicated exactly* at inference (same scaler, same one-hot column order) or predictions silently break.
- **Ensemble methods in practice** — Voting vs. Stacking, and why combining diverse base learners under a meta-estimator can beat any single tuned model.
- **Systematic hyperparameter tuning** with `GridSearchCV` and fair comparison across algorithms using consistent metrics.
- **Model persistence** — serializing and reloading models/scalers so training and serving stay decoupled.
- **Django fundamentals** — routing, templating, form handling, CSRF — and how to wire an inference step into the request/response cycle.
- **End-to-end ownership** — covering the full path from raw CSV to a clickable prediction.
<br/>

## 🔮 Future Improvements

- 🔌 **Expose a REST API** (Django REST Framework) so the models can power other apps, not just the built-in forms.
- 🛡️ **Input validation & error handling** for missing, out-of-range, or malformed submissions.
- 🐳 **Containerize with Docker** for reproducible, environment-independent deployment.
- 🎨 **Improve UI/UX** — move CSS to static files, add a framework, surface prediction confidence/probabilities, improve accessibility.
- 🗄️ **Persist predictions** to the database for history, analytics, and model monitoring.
- 📈 **Monitoring & retraining pipeline** — track input drift and automate periodic retraining.
- ⚡ **Load models once at startup** (and from a versioned registry) instead of reading `.pkl` files on every request.
<br/>

---
<div align="center">

⭐ If you found this project interesting, consider giving it a star!

</div>
