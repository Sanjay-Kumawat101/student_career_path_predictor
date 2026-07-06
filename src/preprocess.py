
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs('../models', exist_ok=True)


df = pd.read_csv('../data/PS2_Dataset.csv')
df.columns = df.columns.str.strip()





TARGET = 'Suggested Job Role'

NUMERIC_COLS = [
    'Logical quotient rating',
    'hackathons',
    'coding skills rating',
    'public speaking points'
]

BINARY_COLS = [
    'self-learning capability?',
    'Extra-courses did',
    'Taken inputs from seniors or elders',
    'worked in teams ever?',
    'Introvert'
]

ORDINAL_COLS = {
    'reading and writing skills' : ['poor', 'medium', 'excellent'],
    'memory capability score'    : ['poor', 'medium', 'excellent'],
}

NOMINAL_COLS = [
    'certifications',
    'workshops',
    'Interested subjects',
    'interested career area',
    'Type of company want to settle in?',
    'Interested Type of Books',
    'Management or Technical',
    'hard/smart worker',
]


for col in BINARY_COLS:
    df[col] = df[col].str.strip().map({'yes': 1, 'no': 0})


for col, order in ORDINAL_COLS.items():
    mapping = {v: i for i, v in enumerate(order)}
    df[col] = df[col].str.strip().map(mapping)


label_encoders = {}
for col in NOMINAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).str.strip())
    label_encoders[col] = le


target_encoder = LabelEncoder()
df[TARGET] = target_encoder.fit_transform(df[TARGET].astype(str).str.strip())

for i, role in enumerate(target_encoder.classes_):
    count = (df[TARGET] == i).sum()


if df.isnull().sum().sum() > 0:
    df.fillna(0, inplace=True)


FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + list(ORDINAL_COLS.keys()) + NOMINAL_COLS

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()



scaler = StandardScaler()
X[NUMERIC_COLS] = scaler.fit_transform(X[NUMERIC_COLS])


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


train_dist = pd.Series(y_train).value_counts().sort_index()
for idx, count in train_dist.items():
    pass


joblib.dump(label_encoders,  '../models/label_encoders.pkl')
joblib.dump(target_encoder,  '../models/target_encoder.pkl')
joblib.dump(scaler,          '../models/scaler.pkl')
joblib.dump(X_train,         '../models/X_train.pkl')
joblib.dump(X_test,          '../models/X_test.pkl')
joblib.dump(y_train,         '../models/y_train.pkl')
joblib.dump(y_test,          '../models/y_test.pkl')
joblib.dump(FEATURE_COLS,    '../models/feature_cols.pkl')
joblib.dump(BINARY_COLS,     '../models/binary_cols.pkl')
joblib.dump(ORDINAL_COLS,    '../models/ordinal_cols.pkl')
joblib.dump(NOMINAL_COLS,    '../models/nominal_cols.pkl')
