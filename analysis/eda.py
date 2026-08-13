import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/customer_churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove missing values
df = df.dropna()

# Convert Churn to numeric
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print("===== CUSTOMER CHURN EDA =====")

print("\nDataset Shape:")
print(df.shape)

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

print("\nDataset Information:")
print(df.info())

# -------------------------------
# 1. Churn Distribution
# -------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig("models/churn_distribution.png")

plt.show()


# -------------------------------
# 2. Churn by Contract
# -------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("models/churn_by_contract.png")

plt.show()


# -------------------------------
# 3. Churn by Internet Service
# -------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig("models/churn_by_internet_service.png")

plt.show()


# -------------------------------
# 4. Tenure vs Churn
# -------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure Distribution by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Tenure (Months)")

plt.tight_layout()

plt.savefig("models/tenure_vs_churn.png")

plt.show()


# -------------------------------
# 5. Monthly Charges vs Churn
# -------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig("models/monthly_charges_vs_churn.png")

plt.show()

print("\nEDA completed successfully!")
print("Charts saved inside the models folder.")