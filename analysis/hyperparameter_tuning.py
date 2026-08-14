import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("data/customer_churn.csv")


# ==============================
# Data Cleaning
# ==============================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()

df = df.drop(
    columns=["customerID"]
)

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# ==============================
# Features & Target
# ==============================

X = df.drop(columns=["Churn"])
y = df["Churn"]


# ==============================
# Feature Types
# ==============================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns


# ==============================
# Preprocessor
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ==============================
# Pipeline
# ==============================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000
            )
        )
    ]
)


# ==============================
# Train/Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# Hyperparameter Grid
# ==============================

param_grid = {
    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "model__solver": [
        "liblinear",
        "lbfgs"
    ],

    "model__class_weight": [
        None,
        "balanced"
    ]
}


# ==============================
# Grid Search
# ==============================

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)


# ==============================
# Best Parameters
# ==============================

print("\n===== BEST PARAMETERS =====")

print(
    grid_search.best_params_
)


print("\nBest Cross-Validation F1 Score:")

print(
    grid_search.best_score_
)


# ==============================
# Test Evaluation
# ==============================

best_model = grid_search.best_estimator_

predictions = best_model.predict(
    X_test
)

probabilities = best_model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n===== TUNED MODEL PERFORMANCE =====")

print(
    f"Accuracy:  {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 Score:  {f1:.4f}"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)