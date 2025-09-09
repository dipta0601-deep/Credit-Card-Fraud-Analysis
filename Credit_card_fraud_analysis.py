# ============================================================
# Credit Card Fraud Detection - Exploratory Data Analysis
# ============================================================

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1️ Load Dataset
# ============================================================
df = pd.read_csv(r"F:\All Excel Practice Files\creditcard.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum().sum())


# ============================================================
# 2️ Data Cleaning
# ============================================================
# Check duplicates
duplicates = df.duplicated().sum()
print("\nNumber of duplicate rows:", duplicates)

# Drop duplicates if any
df.drop_duplicates(inplace=True)


# ============================================================
# 3️ Feature Engineering
# ============================================================
# Fraud / Non-Fraud Label
df["Fraudilent"] = df["Class"].apply(lambda x: "Non_Fraud" if x == 0 else "Fraud")

# Scaling Amount (Min-Max Normalization)
df["Amount-Scaled"] = (df["Amount"] - df["Amount"].min()) / (df["Amount"].max() - df["Amount"].min())

# Log Transformation of Amount
df["Amount_Log"] = np.log1p(df["Amount"])

# Converting Time into Hours & Days
df["Time Hours"] = df["Time"] / 3600
df["Time Days"] = df["Time"] / 86400


# ============================================================
# 4️ Outlier Detection (IQR method on Amount)
# ============================================================
Q1 = df["Amount"].quantile(0.25)
Q3 = df["Amount"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["Amount"] < lower_bound) | (df["Amount"] > upper_bound)]
print("\nNumber of outliers in Amount:", outliers.shape[0])


# ============================================================
# 5️ Data Visualization
# ============================================================

# ----- Histogram: Distribution of Amount -----
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(df['Amount'], bins=30, color='skyblue', edgecolor='black')

# Annotate counts above bars
for i in range(len(counts)):
    plt.text((bins[i] + bins[i+1]) / 2, counts[i], str(int(counts[i])),
             ha='center', va='bottom', fontsize=8)

plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Transaction Amounts')
plt.grid(alpha=0.4)
plt.show()


# ----- Bar Chart: Fraud vs Non-Fraud -----
class_counts = df["Class"].value_counts()

plt.figure(figsize=(6,4))
ax = class_counts.plot(kind="bar", color=["skyblue", "yellow"], edgecolor="black")

# Annotate counts
for i, count in enumerate(class_counts):
    plt.text(i, count + count*0.01, str(int(count)), ha="center", va="bottom", fontsize=10)

plt.title("Fraud vs Non-Fraud Transaction Counts")
plt.xlabel("Transaction Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()


# ----- Correlation Heatmap -----
columns = [col for col in df.columns if col.startswith("V")] + ["Amount", "Class"]

plt.figure(figsize=(16,12))
sns.heatmap(data=df[columns].corr(), cmap="coolwarm", annot=False, linewidths=0.6)
plt.title("Correlation Heatmap of V-Features, Amount, and Class")
plt.show()


# ----- Fraud Frequency Over Time -----
fraud_over_time = df[df["Class"] == 1].groupby("Time Hours").size()

plt.figure(figsize=(12,6))
plt.plot(fraud_over_time.index, fraud_over_time.values, marker="o", linestyle="-", color="red")

plt.title("Fraud Frequency Over Time (by Hour)")
plt.xlabel("Time (Hours)")
plt.ylabel("Number of Fraudulent Transactions")
plt.grid(True, alpha=0.4)
plt.show()

# ============================================================
# 6️ Summary
# ============================================================
print(" Summary of Findings:")
print("- Dataset is highly imbalanced: Fraud cases are extremely rare.")
print("- Transaction amounts are heavily skewed with outliers.")
print("- Feature engineering created new insights: Amount scaling, log transformation, and time-based features.")
print("- Fraud cases tend to cluster in certain time intervals.")
print("- Heatmap reveals correlations between some V-features and fraud class.")