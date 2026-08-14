import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
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
# Precision-Recall Curves
# ==============================

plt.figure(figsize=(9, 6))


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

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    average_precision = average_precision_score(
        y_test,
        probabilities
    )

    plt.plot(
        recall,
        precision,
        label=f"{name} (AP = {average_precision:.3f})"
    )


plt.title(
    "Precision-Recall Curve - Customer Churn Prediction"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "models/precision_recall_curve.png"
)

plt.show()


print("\nPrecision-Recall curve analysis completed!")
print(
    "Chart saved to models/precision_recall_curve.png"
)