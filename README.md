# Student Career Path Predictor

ML-powered project that predicts a student's suggested job role based on a dataset of learner attributes. The project includes data preprocessing, model training (multiple ML models + a TensorFlow model), and a Streamlit web app for interactive predictions.

**Live demo**

The Streamlit app is deployed and available at: https://studentcareerpathpredictor.streamlit.app/

**Project layout**

- `app/` — Streamlit app (`streamlit_app.py`).
- `data/` — raw dataset (`PS2_Dataset.csv`).
- `models/` — saved model artifacts produced by training scripts.
- `notebooks/` — exploratory analysis notebooks.
- `src/` — preprocessing and training scripts (`preprocess.py`, `train_models.py`, `setup_models.py`, `recommend.py`).
- `requirements.txt` — trimmed, project-specific Python dependencies.

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
pip install -r requirements.txt
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

