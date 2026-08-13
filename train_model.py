import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Load dataset
df = pd.read_csv("data/customer_churn.csv")

print("Original Shape:", df.shape)

# Remove customer ID because it does not help predict churn
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values created during conversion
print("\nMissing Values After Conversion:")
print(df.isnull().sum())

# Remove rows with missing TotalCharges
df.dropna(inplace=True)

print("\nShape After Cleaning:", df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Convert target into numbers
y = y.map({
    "Yes": 1,
    "No": 0
})

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumerical Columns:")
print(numerical_columns)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        )
    ]
)

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the same preprocessing
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Data:", X_train_processed.shape)
print("Processed Testing Data:", X_test_processed.shape)

# Create Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
# Create Random Forest model
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train Random Forest
random_forest_model.fit(
    X_train_processed,
    y_train
)

print("\nRandom Forest model trained successfully!")

# Make predictions
rf_pred = random_forest_model.predict(
    X_test_processed
)

# Calculate metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

rf_probability = random_forest_model.predict_proba(
    X_test_processed
)[:, 1]

rf_roc_auc = roc_auc_score(
    y_test,
    rf_probability
)

print("\n===== RANDOM FOREST EVALUATION =====")

print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"F1 Score : {rf_f1:.4f}")
print(f"ROC-AUC  : {rf_roc_auc:.4f}")

print("\n===== RANDOM FOREST CLASSIFICATION REPORT =====")
print(classification_report(y_test, rf_pred))

print("\n===== RANDOM FOREST CONFUSION MATRIX =====")
print(confusion_matrix(y_test, rf_pred))

# Plot Random Forest confusion matrix
rf_cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Churn", "Churn"],
    yticklabels=["Not Churn", "Churn"]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nConfusion matrix saved successfully!")
# Train the model
logistic_model.fit(
    X_train_processed,
    y_train
)

print("\nLogistic Regression model trained successfully!")

# Make predictions
y_pred = logistic_model.predict(X_test_processed)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ROC-AUC requires probability predictions
y_probability = logistic_model.predict_proba(
    X_test_processed
)[:, 1]

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n===== MODEL EVALUATION =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

print("\nFirst 10 Predictions:")
print(y_pred[:10])

# Compare models
print("\n===== MODEL COMPARISON =====")

print(f"Logistic Regression F1 : {f1:.4f}")
print(f"Random Forest F1       : {rf_f1:.4f}")

if rf_f1 >= f1:
    best_model = random_forest_model
    best_model_name = "Random Forest"
    best_f1 = rf_f1
else:
    best_model = logistic_model
    best_model_name = "Logistic Regression"
    best_f1 = f1

print(f"\nBest Model: {best_model_name}")
print(f"Best F1 Score: {best_f1:.4f}")

# Save the best model
joblib.dump(
    best_model,
    "models/churn_model.pkl"
)

# Save the preprocessing pipeline
joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print("\nModel saved successfully!")
print("Preprocessor saved successfully!")