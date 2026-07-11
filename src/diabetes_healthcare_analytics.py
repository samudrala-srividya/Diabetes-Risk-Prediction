"""
Diabetes Healthcare Analytics - End-to-End Project
-------------------------------------------------
This template combines the major stages of a diabetes analytics project:
1. Load data
2. Explore data
3. Clean data
4. Visualize
5. Train ML models
6. Evaluate models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("diabetes.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# -----------------------------
# Data Cleaning
# -----------------------------
cols_with_invalid_zero = [
    "Glucose","BloodPressure","SkinThickness","Insulin","BMI"
]

for col in cols_with_invalid_zero:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Exploratory Data Analysis
# -----------------------------
print("\nOutcome Distribution")
print(df["Outcome"].value_counts())

df.hist(figsize=(10,8))
plt.tight_layout()
plt.show()

corr = df.corr(numeric_only=True)
plt.figure(figsize=(8,6))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# -----------------------------
# Prepare Data
# -----------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {"Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, proba)
    else:
        auc = None

    acc = accuracy_score(y_test, pred)
    cv = cross_val_score(model, X, y, cv=5).mean()

    results.append((name, acc, cv, auc))

    print("\n==============================")
    print(name)
    print("==============================")
    print("Accuracy:", round(acc,4))
    print("Cross Validation:", round(cv,4))
    if auc is not None:
        print("ROC AUC:", round(auc,4))
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test,pred))
    print("\nClassification Report")
    print(classification_report(y_test,pred))

results_df = pd.DataFrame(
    results,
    columns=["Model","Accuracy","Cross Validation","ROC AUC"]
)

print("\nModel Comparison")
print(results_df.sort_values("Accuracy", ascending=False))

best = results_df.sort_values("Accuracy", ascending=False).iloc[0]
print("\nBest Model:")
print(best)
