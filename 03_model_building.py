#          PHASE 6 : DATA PREPROCESSING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Create Output Folders

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/graphs", exist_ok=True)
# Load Engineered Dataset

print("="*80)
print("PHASE 6 : DATA PREPROCESSING")
print("="*80)



df = pd.read_csv(
    "outputs/engineered_data/engineered_bank_churn.csv"
)

print("\nDataset Loaded Successfully")

print(df.head())

print("\nDataset Shape :", df.shape)
# Remove Target Variable


X = df.drop("Exited", axis=1)

y = df["Exited"]

print("\nFeature Matrix Shape :", X.shape)

print("Target Shape :", y.shape)
# One Hot Encoding


print("\nPerforming One Hot Encoding...")

categorical_columns = X.select_dtypes(include=["object","category"]).columns

print("\nCategorical Columns")

print(categorical_columns)

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print("\nEncoding Completed")

print("Encoded Shape :", X.shape)
# Feature Names

feature_names = X.columns.tolist()

feature_df = pd.DataFrame({

    "Features": feature_names

})

feature_df.to_csv(

    "outputs/reports/model_features.csv",

    index=False

)

print("\nFeature List Saved")
# Train Test Split

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Shape")

print(X_train.shape)

print("\nTesting Shape")

print(X_test.shape)
# Target Distribution


print("\nTraining Target Distribution")

print(y_train.value_counts())

print("\nTesting Target Distribution")

print(y_test.value_counts())
# Feature Scaling

print("\nApplying Standard Scaling...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Scaling Completed")
# Save Scaler


joblib.dump(

    scaler,

    "outputs/models/scaler.pkl"

)

print("\nScaler Saved Successfully")
# Convert Back To DataFrame

X_train_scaled = pd.DataFrame(

    X_train_scaled,

    columns=X_train.columns

)

X_test_scaled = pd.DataFrame(

    X_test_scaled,

    columns=X_test.columns

)

# Check Null Values

print("\nChecking Null Values")

print(X_train_scaled.isnull().sum().sum())

print(X_test_scaled.isnull().sum().sum())
# Feature Statistics


print("\nScaled Feature Summary")

print(X_train_scaled.describe())

# Save Processed Dataset


X_train_scaled.to_csv(

    "outputs/models/X_train.csv",

    index=False

)

X_test_scaled.to_csv(

    "outputs/models/X_test.csv",

    index=False

)

y_train.to_csv(

    "outputs/models/y_train.csv",

    index=False

)

y_test.to_csv(

    "outputs/models/y_test.csv",

    index=False

)

print("\nProcessed Data Saved")

# Correlation Heatmap

plt.figure(figsize=(15,10))

sns.heatmap(

    X_train_scaled.corr(),

    cmap="coolwarm",

    center=0,

    cbar=True

)

plt.title("Feature Correlation")

plt.tight_layout()

plt.savefig(

    "outputs/graphs/model_feature_correlation.png"

)

plt.show()
# Completion

print("\n" + "="*80)
print("PHASE 6 COMPLETED SUCCESSFULLY")
print("="*80)

print("\nDataset Ready For Machine Learning Models")
# PHASE 7 : MODEL DEVELOPMENT

print("\n" + "="*80)
print("PHASE 7 : MODEL DEVELOPMENT")
print("="*80)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

models = {

    "Logistic Regression": LogisticRegression(
        random_state=42,
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )

}

results = []

trained_models = {}

print("\nTraining Models...\n")

for name, model in models.items():

    print("="*70)
    print(f"Training {name}")

    if name == "Logistic Regression":

        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

        probabilities = model.predict_proba(X_test_scaled)[:,1]

    else:

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:,1]

    trained_models[name] = model

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc:.4f}")

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc
    ])

# Model Comparison

comparison = pd.DataFrame(

    results,

    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ]

)

comparison = comparison.sort_values(
    by="ROC AUC",
    ascending=False
)

print("\nModel Comparison")
print(comparison)

comparison.to_csv(
    "outputs/reports/model_comparison.csv",
    index=False
)
# Best Model

best_model_name = comparison.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model :", best_model_name)

joblib.dump(
    best_model,
    "outputs/models/best_model.pkl"
)

print("\nBest Model Saved Successfully")

for name, model in trained_models.items():

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        f"outputs/models/{filename}"
    )

print("\nAll Models Saved Successfully")

print("\n" + "="*80)
print("PHASE 7 COMPLETED")
print("="*80)
# PHASE 8 : HYPERPARAMETER TUNING

print("\n" + "="*80)
print("PHASE 8 : HYPERPARAMETER TUNING")
print("="*80)

from sklearn.model_selection import RandomizedSearchCV
# Random Forest Tuning

print("\nTuning Random Forest...")

rf_params = {

    "n_estimators": [100, 200, 300],

    "max_depth": [5, 10, 15, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4],

    "max_features": ["sqrt", "log2"]

}

rf_search = RandomizedSearchCV(

    estimator=RandomForestClassifier(random_state=42),

    param_distributions=rf_params,

    n_iter=20,

    cv=3,

    scoring="roc_auc",

    random_state=42,

    n_jobs=-1,

    verbose=1

)

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_

print("\nBest Random Forest Parameters")

print(rf_search.best_params_)
# Gradient Boosting Tuning

print("\nTuning Gradient Boosting...")

gb_params = {

    "n_estimators": [100, 200],

    "learning_rate": [0.01, 0.05, 0.1],

    "max_depth": [3, 5],

    "subsample": [0.8, 1.0]

}

gb_search = RandomizedSearchCV(

    estimator=GradientBoostingClassifier(random_state=42),

    param_distributions=gb_params,

    n_iter=10,

    cv=3,

    scoring="roc_auc",

    random_state=42,

    n_jobs=-1,

    verbose=1

)

gb_search.fit(X_train, y_train)

best_gb = gb_search.best_estimator_

print("\nBest Gradient Boosting Parameters")

print(gb_search.best_params_)
# Evaluate Tuned Models

models = {

    "Random Forest": best_rf,

    "Gradient Boosting": best_gb

}

results = []

trained_models = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    trained_models[name] = model

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    roc = roc_auc_score(y_test, probabilities)

    results.append([

        name,

        accuracy,

        precision,

        recall,

        f1,

        roc

    ])

comparison = pd.DataFrame(

    results,

    columns=[

        "Model",

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC AUC"

    ]

)

comparison = comparison.sort_values(

    by="ROC AUC",

    ascending=False

)

print("\nTuned Model Comparison")

print(comparison)

comparison.to_csv(

    "outputs/reports/tuned_model_results.csv",

    index=False

)
# Best Tuned Model

best_model_name = comparison.iloc[0]["Model"]

final_model = trained_models[best_model_name]

print("\nBest Tuned Model :", best_model_name)

joblib.dump(

    final_model,

    "outputs/models/final_model.pkl"

)

print("\nFinal Model Saved Successfully")

print("\n" + "="*80)
print("PHASE 8 COMPLETED")
print("="*80)

# PHASE 9 : MODEL EVALUATION

print("\n" + "="*80)
print("PHASE 9 : MODEL EVALUATION")
print("="*80)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
# Prediction

print("\nMaking Predictions...")

y_pred = final_model.predict(X_test)

y_prob = final_model.predict_proba(X_test)[:,1]
# Accuracy Metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\nMODEL PERFORMANCE")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc:.4f}")
# Classification Report

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

report = pd.DataFrame(

    classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

).transpose()

report.to_csv(
    "outputs/reports/classification_report.csv"
)
# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    cmap="Blues",

    fmt="d"

)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(

    "outputs/graphs/confusion_matrix.png"

)

plt.show()

# ROC Curve

plt.figure(figsize=(7,6))

RocCurveDisplay.from_predictions(

    y_test,

    y_prob

)

plt.title("ROC Curve")

plt.savefig(

    "outputs/graphs/roc_curve.png"

)

plt.show()
# Precision Recall Curve

plt.figure(figsize=(7,6))

PrecisionRecallDisplay.from_predictions(

    y_test,

    y_prob

)

plt.title("Precision Recall Curve")

plt.savefig(

    "outputs/graphs/precision_recall_curve.png"

)

plt.show()
# Feature Importance

if hasattr(final_model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": X_train.columns,

        "Importance": final_model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    importance.to_csv(

        "outputs/reports/feature_importance.csv",

        index=False

    )

    plt.figure(figsize=(10,8))

    sns.barplot(

        data=importance.head(15),

        x="Importance",

        y="Feature"

    )

    plt.title("Top 15 Important Features")

    plt.tight_layout()

    plt.savefig(

        "outputs/graphs/feature_importance.png"

    )

    plt.show()
# Final Metrics CSV

metrics = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC AUC"

    ],

    "Value":[

        accuracy,

        precision,

        recall,

        f1,

        roc

    ]

})

metrics.to_csv(

    "outputs/reports/final_model_metrics.csv",

    index=False

)

import os

os.makedirs(
    "outputs/cleaned_data",
    exist_ok=True
)


X.to_csv(
    "outputs/cleaned_data/final_features.csv",
    index=False
)


y.to_csv(
    "outputs/cleaned_data/final_target.csv",
    index=False
)


print("Final Features Saved Successfully")
# Save Final Model


import joblib


joblib.dump(
    final_model,
    "outputs/models/final_model.pkl"
)


print("\nFinal Model Saved Successfully")
# Save Final Model

joblib.dump(

    final_model,

    "outputs/models/final_model.pkl"

)

print("\nFinal Model Saved Successfully")
# Summary
print("\n" + "="*80)
print("FINAL MODEL SUMMARY")
print("="*80)

print(f"""

Best Model : {best_model_name}

Accuracy   : {accuracy:.4f}

Precision  : {precision:.4f}

Recall     : {recall:.4f}

F1 Score   : {f1:.4f}

ROC AUC    : {roc:.4f}

""")

print("="*80)
print("MODEL BUILDING COMPLETED SUCCESSFULLY")
print("="*80)