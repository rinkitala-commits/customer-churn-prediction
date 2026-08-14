import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


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
# Logistic Regression
# ==============================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# ==============================
# Train Model
# ==============================

model.fit(
    X_train,
    y_train
)


# ==============================
# Predictions
# ==============================

predictions = model.predict(
    X_test
)


# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n===== CONFUSION MATRIX =====")
print(cm)


print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Not Churn",
            "Churn"
        ]
    )
)


# ==============================
# Visualization
# ==============================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Not Churn",
        "Churn"
    ],
    yticklabels=[
        "Not Churn",
        "Churn"
    ]
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png"
)

plt.show()


print(
    "\nConfusion matrix saved to "
    "models/confusion_matrix.png"
)