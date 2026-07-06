
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'PS2_Dataset.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
TARGET = 'Suggested Job Role'

NUMERIC_COLS = [
    'Logical quotient rating', 'hackathons',
    'coding skills rating', 'public speaking points'
]
BINARY_COLS = [
    'self-learning capability?', 'Extra-courses did',
    'Taken inputs from seniors or elders',
    'worked in teams ever?', 'Introvert'
]
ORDINAL_MAP = {
    'reading and writing skills': {'poor': 0, 'medium': 1, 'excellent': 2},
    'memory capability score':    {'poor': 0, 'medium': 1, 'excellent': 2},
    'Management or Technical':    {'Management': 0, 'Technical': 1},
    'hard/smart worker':          {'hard worker': 0, 'smart worker': 1},
}
NOMINAL_COLS = [
    'certifications', 'workshops', 'Interested subjects',
    'interested career area', 'Type of company want to settle in?',
    'Interested Type of Books'
]
FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + list(ORDINAL_MAP.keys()) + NOMINAL_COLS

df_enc = df.copy()
label_encoders = {}

for col in BINARY_COLS:
    df_enc[col] = df_enc[col].str.strip().map({'yes': 1, 'no': 0})

for col, mapping in ORDINAL_MAP.items():
    df_enc[col] = df_enc[col].str.strip().map(mapping)

for col in NOMINAL_COLS:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str).str.strip())
    label_encoders[col] = le

target_encoder = LabelEncoder()
df_enc[TARGET] = target_encoder.fit_transform(df_enc[TARGET].str.strip())
class_names    = target_encoder.classes_

scaler = StandardScaler()
df_enc[NUMERIC_COLS] = scaler.fit_transform(df_enc[NUMERIC_COLS])

X = df_enc[FEATURE_COLS].values.astype(np.float32)
y = df_enc[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


models = {
    'Decision Tree'       : DecisionTreeClassifier(random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes'         : GaussianNB(),
    'SVM'                 : SVC(kernel='rbf', probability=True, random_state=42),
    'KNN'                 : KNeighborsClassifier(n_neighbors=5),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    results[name] = {'train': train_acc, 'test': test_acc, 'model': model}

dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X, y)

joblib.dump(dt_full,        os.path.join(MODEL_DIR, 'best_ml_model.pkl'))
joblib.dump(label_encoders, os.path.join(MODEL_DIR, 'label_encoders.pkl'))
joblib.dump(target_encoder, os.path.join(MODEL_DIR, 'target_encoder.pkl'))
joblib.dump(scaler,         os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(FEATURE_COLS,   os.path.join(MODEL_DIR, 'feature_cols.pkl'))
joblib.dump(ORDINAL_MAP,    os.path.join(MODEL_DIR, 'ordinal_map.pkl'))
joblib.dump(BINARY_COLS,    os.path.join(MODEL_DIR, 'binary_cols.pkl'))
joblib.dump(NOMINAL_COLS,   os.path.join(MODEL_DIR, 'nominal_cols.pkl'))
joblib.dump(X_train,        os.path.join(MODEL_DIR, 'X_train.pkl'))
joblib.dump(X_test,         os.path.join(MODEL_DIR, 'X_test.pkl'))
joblib.dump(y_train,        os.path.join(MODEL_DIR, 'y_train.pkl'))
joblib.dump(y_test,         os.path.join(MODEL_DIR, 'y_test.pkl'))
joblib.dump(results,        os.path.join(MODEL_DIR, 'all_results.pkl'))

for f in sorted(os.listdir(MODEL_DIR)):
    size = os.path.getsize(os.path.join(MODEL_DIR, f))
