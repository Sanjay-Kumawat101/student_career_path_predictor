
import numpy as np
import pandas as pd
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing        import LabelEncoder, StandardScaler
from sklearn.model_selection      import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble             import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors            import KNeighborsClassifier
from sklearn.svm                  import SVC
from sklearn.naive_bayes          import GaussianNB
from sklearn.linear_model         import LogisticRegression
from sklearn.tree                 import DecisionTreeClassifier
from sklearn.metrics              import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost   import XGBClassifier
from lightgbm  import LGBMClassifier
from catboost  import CatBoostClassifier
import keras
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('../models', exist_ok=True)


df = pd.read_csv('../data/PS2_Dataset.csv')
df.columns = df.columns.str.strip()

TARGET       = 'Suggested Job Role'
feature_cols = [c for c in df.columns if c != TARGET]

df_enc = df.copy()

BINARY_COLS = [
    'self-learning capability?', 'Extra-courses did',
    'Taken inputs from seniors or elders',
    'worked in teams ever?', 'Introvert'
]
ORDINAL_MAP = {
    'reading and writing skills' : {'poor': 0, 'medium': 1, 'excellent': 2},
    'memory capability score'    : {'poor': 0, 'medium': 1, 'excellent': 2},
    'Management or Technical'    : {'Management': 0, 'Technical': 1},
    'hard/smart worker'          : {'hard worker': 0, 'smart worker': 1},
}
NOMINAL_COLS = [
    'certifications', 'workshops', 'Interested subjects',
    'interested career area', 'Type of company want to settle in?',
    'Interested Type of Books'
]

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
n_classes      = len(class_names)

NUMERIC_COLS = ['Logical quotient rating', 'hackathons',
                'coding skills rating', 'public speaking points']

scaler = StandardScaler()
df_enc[NUMERIC_COLS] = scaler.fit_transform(df_enc[NUMERIC_COLS])

X = df_enc[feature_cols].values.astype(np.float32)
y = df_enc[TARGET].values

joblib.dump(label_encoders,  '../models/label_encoders.pkl')
joblib.dump(target_encoder,  '../models/target_encoder.pkl')
joblib.dump(scaler,          '../models/scaler.pkl')
joblib.dump(feature_cols,    '../models/feature_cols.pkl')

SKF     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

joblib.dump(X_test,  '../models/X_test.pkl')
joblib.dump(y_test,  '../models/y_test.pkl')

ml_models = {
    "Random Forest"       : RandomForestClassifier(n_estimators=200, max_depth=None,
                                                    min_samples_leaf=1, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=None, random_state=42),
    "KNN"                 : KNeighborsClassifier(n_neighbors=3),
    "SVM"                 : SVC(kernel='rbf', C=10, probability=True, random_state=42),
    "Logistic Regression" : LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Naive Bayes"         : GaussianNB(),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=200,
                                                        max_depth=5, random_state=42),
    "XGBoost"             : XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                           use_label_encoder=False,
                                           eval_metric='mlogloss', random_state=42),
    "LightGBM"            : LGBMClassifier(n_estimators=200, max_depth=6,
                                            random_state=42, verbose=-1),
    "CatBoost"            : CatBoostClassifier(iterations=200, depth=6,
                                                random_state=42, verbose=False),
}

results = {}

for name, model in ml_models.items():
    cv_scores = cross_val_score(model, X_train_full, y_train_full,
                                 cv=SKF, scoring='accuracy', n_jobs=-1)
    model.fit(X_train_full, y_train_full)
    y_pred  = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1  = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        'cv_mean'  : cv_scores.mean(),
        'cv_std'   : cv_scores.std(),
        'test_acc' : test_acc,
        'test_f1'  : test_f1,
        'model'    : model
    }

def build_dl_model(n_feat, n_cls):
    inp = keras.Input(shape=(n_feat,))
    x   = layers.Dense(512, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64,  activation='relu')(x)
    out = layers.Dense(n_cls, activation='softmax')(x)
    return keras.Model(inp, out)

dl_model = build_dl_model(X.shape[1], n_classes)
dl_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = dl_model.fit(
    X_train_full, y_train_full,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=64,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15,
                                      restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=7, min_lr=1e-6, verbose=0)
    ],
    verbose=0
)

y_pred_dl = np.argmax(dl_model.predict(X_test, verbose=0), axis=1)
dl_acc = accuracy_score(y_test, y_pred_dl)
dl_f1  = f1_score(y_test, y_pred_dl, average='weighted')

results["Deep Learning"] = {
    'cv_mean': 0, 'cv_std': 0,
    'test_acc': dl_acc, 'test_f1': dl_f1,
    'model': dl_model
}

sorted_res = sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True)

best_name, best_data = sorted_res[0]

best_model = best_data['model']
if best_name == "Deep Learning":
    y_pred_best = y_pred_dl
else:
    y_pred_best = best_model.predict(X_test)

cm          = confusion_matrix(y_test, y_pred_best)
short_names = [c.split()[0] for c in class_names]

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=short_names, yticklabels=short_names)
plt.title(f'Confusion Matrix — {best_name}', fontsize=13, fontweight='bold')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('../models/confusion_matrix.png')
plt.show()

names  = [n for n, _ in sorted_res]
t_accs = [m['test_acc'] for _, m in sorted_res]
t_f1s  = [m['test_f1']  for _, m in sorted_res]
x      = np.arange(len(names))
w      = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
b1 = ax.bar(x - w/2, t_accs, w, label='Test Accuracy', color='steelblue', alpha=0.85)
b2 = ax.bar(x + w/2, t_f1s,  w, label='F1 Score',      color='coral',     alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
ax.set_title('Model Comparison — Test Accuracy & F1', fontsize=13, fontweight='bold')
ax.legend()
ax.bar_label(b1, fmt='%.3f', fontsize=8, padding=2)
ax.bar_label(b2, fmt='%.3f', fontsize=8, padding=2)
plt.tight_layout()
plt.savefig('../models/model_comparison.png')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('DL — Accuracy', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].legend()

axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('DL — Loss', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].legend()
plt.tight_layout()
plt.savefig('../models/dl_training_history.png')
plt.show()

if best_name == "Deep Learning":
    dl_model.save('../models/best_model.keras')
else:
    joblib.dump(best_model, '../models/best_ml_model.pkl')

joblib.dump(best_name, '../models/best_model_name.pkl')
dl_model.save('../models/dl_model.keras')
