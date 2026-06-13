# ============================================================
# TASK 4: Disease Prediction from Medical Data - CodeAlpha
# ============================================================
# Objective: Predict possibility of diseases from patient data
# Approach: Classification techniques on structured medical data
# Algorithms: SVM, Logistic Regression, Random Forest, XGBoost
# Datasets: Heart Disease, Diabetes, Breast Cancer (UCI ML Repo)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed — using GradientBoosting instead.")

print("=" * 60)
print("TASK 4: DISEASE PREDICTION FROM MEDICAL DATA")
print("=" * 60)

# ─────────────────────────────────────────
# DATASET LOADER FUNCTIONS
# ─────────────────────────────────────────

def load_heart_disease_dataset():
    """Simulate a Heart Disease dataset (UCI format)."""
    np.random.seed(42)
    n = 303
    df = pd.DataFrame({
        'age':       np.random.randint(29, 77, n),
        'sex':       np.random.randint(0, 2, n),
        'cp':        np.random.randint(0, 4, n),          # chest pain type
        'trestbps':  np.random.randint(94, 200, n),       # resting BP
        'chol':      np.random.randint(126, 564, n),      # cholesterol
        'fbs':       np.random.randint(0, 2, n),          # fasting blood sugar
        'restecg':   np.random.randint(0, 3, n),
        'thalach':   np.random.randint(71, 202, n),       # max heart rate
        'exang':     np.random.randint(0, 2, n),          # exercise angina
        'oldpeak':   np.round(np.random.uniform(0, 6.2, n), 1),
        'slope':     np.random.randint(0, 3, n),
        'ca':        np.random.randint(0, 5, n),
        'thal':      np.random.randint(0, 4, n),
    })
    # Label: heart disease present (1) or not (0)
    risk = (df['age']/77*0.2 + (1-df['sex'])*0.05 + df['cp']/4*0.15 +
            df['trestbps']/200*0.1 + df['chol']/564*0.1 + df['exang']*0.15 +
            df['oldpeak']/6.2*0.15 + df['ca']/4*0.1)
    df['target'] = (risk > risk.median()).astype(int)
    return df, 'Heart Disease'


def load_diabetes_dataset():
    """Simulate the Pima Indians Diabetes dataset."""
    np.random.seed(123)
    n = 768
    df = pd.DataFrame({
        'pregnancies':          np.random.randint(0, 17, n),
        'glucose':              np.random.randint(44, 199, n),
        'blood_pressure':       np.random.randint(24, 122, n),
        'skin_thickness':       np.random.randint(7, 99, n),
        'insulin':              np.random.randint(14, 846, n),
        'bmi':                  np.round(np.random.uniform(18.2, 67.1, n), 1),
        'diabetes_pedigree':    np.round(np.random.uniform(0.08, 2.42, n), 3),
        'age':                  np.random.randint(21, 81, n),
    })
    risk = (df['glucose']/199*0.35 + df['bmi']/67.1*0.2 +
            df['age']/81*0.15 + df['insulin']/846*0.1 +
            df['pregnancies']/17*0.1 + df['diabetes_pedigree']/2.42*0.1)
    df['target'] = (risk > risk.median()).astype(int)
    return df, 'Diabetes'


def load_breast_cancer_dataset():
    """Load sklearn's built-in Breast Cancer dataset."""
    bc = load_breast_cancer()
    df = pd.DataFrame(bc.data, columns=bc.feature_names)
    df['target'] = bc.target   # 1=benign, 0=malignant
    return df, 'Breast Cancer'


# ─────────────────────────────────────────
# 1. SELECT & LOAD DATASET
#    Change 'breast_cancer' → 'heart' or 'diabetes' as needed
# ─────────────────────────────────────────
DATASET = 'breast_cancer'   # options: 'breast_cancer', 'heart', 'diabetes'

if DATASET == 'heart':
    df, dataset_name = load_heart_disease_dataset()
elif DATASET == 'diabetes':
    df, dataset_name = load_diabetes_dataset()
else:
    df, dataset_name = load_breast_cancer_dataset()

print(f"\nDataset       : {dataset_name}")
print(f"Shape         : {df.shape}")
print(f"Target balance:\n{df['target'].value_counts()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum().sum()} total")

# ─────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────
fig_eda, axes_eda = plt.subplots(1, 2, figsize=(12, 4))
fig_eda.suptitle(f'Task 4 EDA: {dataset_name}', fontsize=14, fontweight='bold')

# Class distribution
df['target'].value_counts().plot(kind='bar', ax=axes_eda[0],
                                  color=['steelblue','coral'])
axes_eda[0].set_title('Class Distribution')
axes_eda[0].set_xlabel('Class'); axes_eda[0].set_ylabel('Count')
axes_eda[0].tick_params(axis='x', rotation=0)

# Correlation heatmap (top features only)
top_cols = df.corr()['target'].abs().nlargest(11).index.tolist()
corr_sub = df[top_cols].corr()
sns.heatmap(corr_sub, annot=False, cmap='coolwarm', ax=axes_eda[1],
            linewidths=0.5)
axes_eda[1].set_title('Feature Correlation (Top 10 + target)')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/task4_eda.png', dpi=150, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING & SPLIT
# ─────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────
# 4. DEFINE MODELS
# ─────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM':                 SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost' if XGB_AVAILABLE else 'Gradient Boosting':
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        if XGB_AVAILABLE else
        GradientBoostingClassifier(n_estimators=200, random_state=42),
}

# ─────────────────────────────────────────
# 5. TRAIN & EVALUATE
# ─────────────────────────────────────────
results = {}
print(f"\n{'='*60}")
print("MODEL EVALUATION")
print(f"{'='*60}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='roc_auc')

    results[name] = {
        'model':    model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision':precision_score(y_test, y_pred, zero_division=0),
        'recall':   recall_score(y_test, y_pred, zero_division=0),
        'f1':       f1_score(y_test, y_pred, zero_division=0),
        'roc_auc':  roc_auc_score(y_test, y_proba),
        'cv_auc':   cv_scores.mean(),
        'y_pred':   y_pred,
        'y_proba':  y_proba,
    }

    print(f"\n{name}:")
    print(f"  Accuracy    : {results[name]['accuracy']:.4f}")
    print(f"  Precision   : {results[name]['precision']:.4f}")
    print(f"  Recall      : {results[name]['recall']:.4f}")
    print(f"  F1-Score    : {results[name]['f1']:.4f}")
    print(f"  ROC-AUC     : {results[name]['roc_auc']:.4f}")
    print(f"  CV-AUC(5-fold): {results[name]['cv_auc']:.4f}")

# ─────────────────────────────────────────
# 6. FINAL VISUALISATIONS
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Task 4: Disease Prediction — {dataset_name}',
             fontsize=15, fontweight='bold')

# Confusion matrices (first 3 models)
for idx, (name, res) in enumerate(list(results.items())[:3]):
    ax = axes[0][idx]
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
    ax.set_title(f'{name}')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

# ROC Curves
ax_roc = axes[1][0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax_roc.plot(fpr, tpr, label=f"{name} ({res['roc_auc']:.3f})")
ax_roc.plot([0,1],[0,1],'k--', label='Random')
ax_roc.set_title('ROC Curves (All Models)')
ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
ax_roc.legend(fontsize=8)

# Metric bar comparison
ax_bar = axes[1][1]
metric_names = ['accuracy','precision','recall','f1','roc_auc']
x = np.arange(len(metric_names))
width = 0.2
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metric_names]
    ax_bar.bar(x + i*width, vals, width, label=name)
ax_bar.set_xticks(x + width*1.5)
ax_bar.set_xticklabels(['Acc','Pre','Rec','F1','AUC'], fontsize=9)
ax_bar.set_ylim(0, 1.15)
ax_bar.set_title('Metric Comparison')
ax_bar.legend(fontsize=7)

# Feature importance (best tree model)
ax_fi = axes[1][2]
rf_res = results.get('Random Forest', list(results.values())[2])
rf_model = rf_res['model']
if hasattr(rf_model, 'feature_importances_'):
    fi = pd.Series(rf_model.feature_importances_,
                   index=X.columns).nlargest(15).sort_values()
    fi.plot(kind='barh', ax=ax_fi, color='tomato')
    ax_fi.set_title('Top 15 Feature Importances')
else:
    ax_fi.axis('off')
    ax_fi.text(0.3, 0.5, 'Feature importance\nnot available for SVM',
               transform=ax_fi.transAxes, fontsize=12)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/task4_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[Task 4] Results plot saved → task4_results.png")
print("[Task 4] EDA plot saved     → task4_eda.png")

# ─────────────────────────────────────────
# 7. BEST MODEL SUMMARY
# ─────────────────────────────────────────
best = max(results, key=lambda k: results[k]['roc_auc'])
print(f"\n✅ Best model: {best}  (ROC-AUC = {results[best]['roc_auc']:.4f})")
print("\nDetailed Classification Report:")
print(classification_report(y_test, results[best]['y_pred']))

# ─────────────────────────────────────────
# 8. SAVE SUMMARY TABLE
# ─────────────────────────────────────────
summary = pd.DataFrame({
    name: {m: round(res[m], 4) for m in ['accuracy','precision','recall','f1','roc_auc','cv_auc']}
    for name, res in results.items()
}).T
print(f"\nSummary Table:\n{summary.to_string()}")
summary.to_csv('/mnt/user-data/outputs/task4_model_summary.csv')
print("\n[Task 4] Summary table saved → task4_model_summary.csv")
