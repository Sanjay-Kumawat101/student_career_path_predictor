# Student Career Path Predictor

ML-powered project that predicts a student's suggested job role based on a dataset of learner attributes. The project includes data preprocessing, model training (multiple ML models + a TensorFlow model), and a Streamlit web app for interactive predictions.

**Project layout**

- `app/` — Streamlit app (`streamlit_app.py`).
- `data/` — raw dataset (`PS2_Dataset.csv`).
- `models/` — saved model artifacts produced by training scripts.
- `notebooks/` — exploratory analysis notebooks.
- `src/` — preprocessing and training scripts (`preprocess.py`, `train_models.py`, `setup_models.py`, `recommend.py`).
- `requirements.txt/requirements.txt` — trimmed, project-specific Python dependencies.

**Quick start (recommended)**

1. Create a virtual environment (recommended Python 3.10/3.11):

```bash
python -m venv .venv
```

2. Activate it:

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
source .venv/bin/activate
```

3. Install minimal project dependencies:

```bash
pip install -r requirements.txt/requirements.txt
```

4. Prepare artifacts (preprocessing + training):

```bash
python src/preprocess.py      # generates encoders, scaler, train/test splits
python src/train_models.py    # trains models and saves best model(s) into models/
# Or run src/setup_models.py to produce baseline artifacts
```

5. Run the Streamlit app:

```bash
python -m streamlit run app/streamlit_app.py
```

**Notes & recommendations**

- Use Python 3.10 or 3.11 for best compatibility with packages like TensorFlow, XGBoost, LightGBM and CatBoost. The workspace had Python 3.14 detected — those packages may not support 3.14.
- `requirements.txt/requirements.txt` contains a trimmed list of packages used by the project. Add packages back only if you encounter import errors.
- Model artifacts are stored under `models/` after running training scripts. The Streamlit app expects `best_ml_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, `target_encoder.pkl`, and `feature_cols.pkl` (and related artifact files).

**License & attribution**

This repository contains only example code and dataset scaffolding for educational/demo use.
