import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

X = df.drop(
    columns=["Churn"]
)

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
# Preprocessing
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
# Models
# ==============================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}


# ==============================
# Model Evaluation
# ==============================

results = []


for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
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

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc
    })


# ==============================
# Results
# ==============================

results_df = pd.DataFrame(
    results
)

print("\n===== MODEL COMPARISON =====")

print(
    results_df.to_string(
        index=False
    )
)


# ==============================
# Save Results
# ==============================

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print(
    "\nModel comparison saved to "
    "models/model_comparison.csv"
)