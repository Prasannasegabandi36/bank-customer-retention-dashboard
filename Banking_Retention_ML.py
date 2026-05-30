# ============================================================
# Bank Customer Retention Intelligence Dashboard
# Full Machine Learning Notebook Code
# File Name: Banking_Retention_ML.ipynb
# Dataset: European_Bank.csv
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

import warnings
warnings.filterwarnings("ignore")

print("Libraries imported successfully.")


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("European_Bank.csv")

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)

display(df.head())


# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nStatistical Summary:")
display(df.describe())


# ============================================================
# 4. TARGET VARIABLE EXPLANATION
# ============================================================

print("\nTarget Column: Exited")
print("0 = Retained Customer")
print("1 = Churned Customer")

print("\nChurn Distribution:")
print(df["Exited"].value_counts())

churn_rate = df["Exited"].mean() * 100
print(f"\nOverall Churn Rate: {churn_rate:.2f}%")


# ============================================================
# 5. DATA CLEANING
# ============================================================

bank_df = df.copy()

drop_cols = ["RowNumber", "CustomerId", "Surname"]

for col in drop_cols:
    if col in bank_df.columns:
        bank_df.drop(col, axis=1, inplace=True)

print("\nColumns after cleaning:")
print(bank_df.columns.tolist())

display(bank_df.head())


# ============================================================
# 6. CHURN DISTRIBUTION VISUALIZATION
# ============================================================

plt.figure(figsize=(6, 4))
sns.countplot(data=bank_df, x="Exited")
plt.title("Customer Churn Distribution")
plt.xlabel("Exited: 0 = Retained, 1 = Churned")
plt.ylabel("Customer Count")
plt.show()


# ============================================================
# 7. CHURN PERCENTAGE PIE CHART
# ============================================================

churn_counts = bank_df["Exited"].value_counts()
labels = ["Retained", "Churned"]

plt.figure(figsize=(6, 6))
plt.pie(churn_counts, labels=labels, autopct="%1.1f%%", startangle=90)
plt.title("Retained vs Churned Customers")
plt.show()


# ============================================================
# 8. CHURN BY GEOGRAPHY
# ============================================================

geo_churn = bank_df.groupby("Geography")["Exited"].mean().reset_index()
geo_churn["Churn Rate (%)"] = geo_churn["Exited"] * 100

plt.figure(figsize=(7, 4))
sns.barplot(data=geo_churn, x="Geography", y="Churn Rate (%)")
plt.title("Churn Rate by Geography")
plt.ylabel("Churn Rate (%)")
plt.show()

display(geo_churn)


# ============================================================
# 9. CHURN BY GENDER
# ============================================================

gender_churn = bank_df.groupby("Gender")["Exited"].mean().reset_index()
gender_churn["Churn Rate (%)"] = gender_churn["Exited"] * 100

plt.figure(figsize=(6, 4))
sns.barplot(data=gender_churn, x="Gender", y="Churn Rate (%)")
plt.title("Churn Rate by Gender")
plt.ylabel("Churn Rate (%)")
plt.show()

display(gender_churn)


# ============================================================
# 10. AGE DISTRIBUTION BY CHURN
# ============================================================

plt.figure(figsize=(8, 5))
sns.histplot(data=bank_df, x="Age", hue="Exited", kde=True, bins=30)
plt.title("Age Distribution by Churn")
plt.xlabel("Age")
plt.ylabel("Customer Count")
plt.show()


# ============================================================
# 11. BALANCE DISTRIBUTION BY CHURN
# ============================================================

plt.figure(figsize=(8, 5))
sns.histplot(data=bank_df, x="Balance", hue="Exited", kde=True, bins=30)
plt.title("Balance Distribution by Churn")
plt.xlabel("Balance")
plt.ylabel("Customer Count")
plt.show()


# ============================================================
# 12. ACTIVE MEMBERSHIP VS CHURN
# ============================================================

active_churn = bank_df.groupby("IsActiveMember")["Exited"].mean().reset_index()
active_churn["Churn Rate (%)"] = active_churn["Exited"] * 100
active_churn["Customer Type"] = active_churn["IsActiveMember"].map({
    0: "Inactive Customer",
    1: "Active Customer"
})

plt.figure(figsize=(6, 4))
sns.barplot(data=active_churn, x="Customer Type", y="Churn Rate (%)")
plt.title("Active vs Inactive Customer Churn")
plt.ylabel("Churn Rate (%)")
plt.show()

display(active_churn)


# ============================================================
# 13. NUMBER OF PRODUCTS VS CHURN
# ============================================================

product_churn = bank_df.groupby("NumOfProducts")["Exited"].mean().reset_index()
product_churn["Churn Rate (%)"] = product_churn["Exited"] * 100

plt.figure(figsize=(7, 4))
sns.barplot(data=product_churn, x="NumOfProducts", y="Churn Rate (%)")
plt.title("Number of Products vs Churn Rate")
plt.xlabel("Number of Products")
plt.ylabel("Churn Rate (%)")
plt.show()

display(product_churn)


# ============================================================
# 14. CORRELATION HEATMAP
# ============================================================

numeric_df = bank_df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# ============================================================
# 15. FEATURE ENGINEERING
# ============================================================

bank_df["Engagement_Profile"] = np.select(
    [
        (bank_df["IsActiveMember"] == 1) & (bank_df["NumOfProducts"] >= 2),
        (bank_df["IsActiveMember"] == 0) & (bank_df["Balance"] > 100000),
        (bank_df["IsActiveMember"] == 1) & (bank_df["NumOfProducts"] == 1),
        (bank_df["IsActiveMember"] == 0)
    ],
    [
        "Active Engaged Customer",
        "Inactive High-Balance Customer",
        "Active Low-Product Customer",
        "Inactive Disengaged Customer"
    ],
    default="Other Customer"
)

# RSI = Relationship Strength Index
bank_df["RSI"] = (
    np.where(bank_df["IsActiveMember"] == 1, 1, 0) +
    np.where(bank_df["NumOfProducts"] >= 2, 1, 0) +
    np.where(bank_df["HasCrCard"] == 1, 1, 0) +
    np.where(bank_df["Tenure"] >= 5, 1, 0)
)

bank_df["Relationship_Category"] = pd.cut(
    bank_df["RSI"],
    bins=[-1, 1, 2, 4],
    labels=["Weak Relationship", "Medium Relationship", "Strong Relationship"]
)

bank_df["Product_Group"] = np.select(
    [
        bank_df["NumOfProducts"] == 1,
        bank_df["NumOfProducts"] == 2,
        bank_df["NumOfProducts"] >= 3
    ],
    [
        "Single Product Customer",
        "Two Product Customer",
        "Multi Product Customer"
    ],
    default="Other"
)

bank_df["SalaryBalanceMismatch"] = np.where(
    (bank_df["EstimatedSalary"] > bank_df["EstimatedSalary"].median()) &
    (bank_df["Balance"] == 0),
    1,
    0
)

bank_df["Retention_Intelligence_Score"] = (
    bank_df["RSI"] * 20 +
    bank_df["IsActiveMember"] * 20 +
    np.where(bank_df["Balance"] > 100000, 10, 0)
)

bank_df["Risk_Level"] = "High Risk"

bank_df.loc[
    bank_df["Retention_Intelligence_Score"] >= 80,
    "Risk_Level"
] = "Low Risk"

bank_df.loc[
    (bank_df["Retention_Intelligence_Score"] >= 50) &
    (bank_df["Retention_Intelligence_Score"] < 80),
    "Risk_Level"
] = "Medium Risk"

print("\nFeature engineering completed successfully.")

display(bank_df.head())


# ============================================================
# 16. ENGAGEMENT PROFILE ANALYSIS
# ============================================================

engagement_churn = bank_df.groupby("Engagement_Profile")["Exited"].mean().reset_index()
engagement_churn["Churn Rate (%)"] = engagement_churn["Exited"] * 100

plt.figure(figsize=(10, 5))
sns.barplot(data=engagement_churn, x="Engagement_Profile", y="Churn Rate (%)")
plt.title("Churn Rate by Engagement Profile")
plt.xticks(rotation=30, ha="right")
plt.ylabel("Churn Rate (%)")
plt.show()

display(engagement_churn)


# ============================================================
# 17. RELATIONSHIP STRENGTH ANALYSIS
# ============================================================

relationship_churn = bank_df.groupby("Relationship_Category")["Exited"].mean().reset_index()
relationship_churn["Churn Rate (%)"] = relationship_churn["Exited"] * 100

plt.figure(figsize=(7, 4))
sns.barplot(data=relationship_churn, x="Relationship_Category", y="Churn Rate (%)")
plt.title("Relationship Strength vs Churn")
plt.ylabel("Churn Rate (%)")
plt.show()

display(relationship_churn)


# ============================================================
# 18. PRODUCT GROUP ANALYSIS
# ============================================================

product_group_churn = bank_df.groupby("Product_Group")["Exited"].mean().reset_index()
product_group_churn["Churn Rate (%)"] = product_group_churn["Exited"] * 100

plt.figure(figsize=(8, 4))
sns.barplot(data=product_group_churn, x="Product_Group", y="Churn Rate (%)")
plt.title("Product Group vs Churn")
plt.ylabel("Churn Rate (%)")
plt.show()

display(product_group_churn)


# ============================================================
# 19. ENCODE CATEGORICAL COLUMNS
# ============================================================

ml_df = bank_df.copy()

le_geo = LabelEncoder()
le_gender = LabelEncoder()

ml_df["Geography_enc"] = le_geo.fit_transform(ml_df["Geography"])
ml_df["Gender_enc"] = le_gender.fit_transform(ml_df["Gender"])

print("\nEncoding completed successfully.")

display(ml_df[["Geography", "Geography_enc", "Gender", "Gender_enc"]].head())


# ============================================================
# 20. SELECT ML FEATURES AND TARGET
# ============================================================

features = [
    "CreditScore",
    "Geography_enc",
    "Gender_enc",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

X = ml_df[features]
y = ml_df["Exited"]

print("\nFeatures used for ML:")
print(features)

print("\nTarget variable: Exited")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 21. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ============================================================
# 22. FEATURE SCALING FOR LOGISTIC REGRESSION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaling completed for Logistic Regression.")


# ============================================================
# 23. TRAIN THREE ML MODELS
# ============================================================

models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "scaled": True
    },
    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        "scaled": False
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(
            n_estimators=150,
            random_state=42
        ),
        "scaled": False
    }
}

results = {}

for name, item in models.items():
    model = item["model"]

    if item["scaled"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Model": model,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Predictions": y_pred,
        "Probabilities": y_prob
    }

print("\nAll models trained successfully.")


# ============================================================
# 24. MODEL COMPARISON TABLE
# ============================================================

model_comparison = pd.DataFrame({
    model_name: {
        "Accuracy": model_result["Accuracy"],
        "Precision": model_result["Precision"],
        "Recall": model_result["Recall"],
        "F1 Score": model_result["F1 Score"],
        "AUC": model_result["AUC"]
    }
    for model_name, model_result in results.items()
}).T

model_comparison = model_comparison.sort_values(by="AUC", ascending=False)

print("\nModel Comparison:")
display(model_comparison)


# ============================================================
# 25. SELECT BEST MODEL
# ============================================================

best_model_name = "Gradient Boosting"
best_model = results[best_model_name]["Model"]

print("\nBest Model Selected:", best_model_name)
print("\nPerformance:")
display(model_comparison.loc[[best_model_name]])


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

y_pred_best = results[best_model_name]["Predictions"]

print("\nClassification Report - Gradient Boosting")
print(classification_report(y_test, y_pred_best))


# ============================================================
# 27. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Retained", "Predicted Churned"],
    yticklabels=["Actual Retained", "Actual Churned"]
)

plt.title("Confusion Matrix - Gradient Boosting")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# 28. ROC CURVE FOR ALL MODELS
# ============================================================

plt.figure(figsize=(8, 6))

for name, item in results.items():
    y_prob = item["Probabilities"]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_value = item["AUC"]
    plt.plot(fpr, tpr, label=f"{name} AUC = {auc_value:.4f}")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# ============================================================
# 29. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(data=feature_importance, x="Importance", y="Feature")
plt.title("Feature Importance - Gradient Boosting")
plt.show()

display(feature_importance)


# ============================================================
# 30. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("\nCross Validation AUC Scores:", cv_scores)
print("Mean CV AUC:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())


# ============================================================
# 31. ADD ML CHURN PROBABILITY TO DATASET
# ============================================================

bank_df["ML_Churn_Probability"] = best_model.predict_proba(X)[:, 1]

bank_df["ML_Churn_Risk"] = pd.cut(
    bank_df["ML_Churn_Probability"],
    bins=[0, 0.33, 0.66, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    include_lowest=True
)

print("\nML churn probability and risk level added successfully.")

display(
    bank_df[
        [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Balance",
            "NumOfProducts",
            "IsActiveMember",
            "Exited",
            "Engagement_Profile",
            "RSI",
            "Relationship_Category",
            "ML_Churn_Probability",
            "ML_Churn_Risk"
        ]
    ].head()
)


# ============================================================
# 32. ML RISK DISTRIBUTION
# ============================================================

risk_distribution = bank_df["ML_Churn_Risk"].value_counts().reset_index()
risk_distribution.columns = ["ML_Churn_Risk", "Customer Count"]

plt.figure(figsize=(7, 4))
sns.barplot(data=risk_distribution, x="ML_Churn_Risk", y="Customer Count")
plt.title("ML Churn Risk Distribution")
plt.ylabel("Customer Count")
plt.show()

display(risk_distribution)


# ============================================================
# 33. TOP 20 HIGH-RISK CUSTOMERS
# ============================================================

top_20_high_risk = bank_df.sort_values(
    by="ML_Churn_Probability",
    ascending=False
).head(20)

print("\nTop 20 High-Risk Customers:")

display(
    top_20_high_risk[
        [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Exited",
            "ML_Churn_Probability",
            "ML_Churn_Risk"
        ]
    ]
)


# ============================================================
# 34. PREMIUM RISK CUSTOMERS
# ============================================================

premium_risk_customers = bank_df[
    (bank_df["Balance"] > 100000) &
    (bank_df["IsActiveMember"] == 0)
].copy()

print("\nPremium Risk Customers Count:", premium_risk_customers.shape[0])

display(
    premium_risk_customers[
        [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Balance",
            "NumOfProducts",
            "IsActiveMember",
            "Exited",
            "ML_Churn_Probability",
            "ML_Churn_Risk"
        ]
    ].head(10)
)


# ============================================================
# 35. LIVE CUSTOMER PREDICTION EXAMPLE
# ============================================================

new_customer = pd.DataFrame({
    "CreditScore": [650],
    "Geography_enc": [le_geo.transform(["Germany"])[0]],
    "Gender_enc": [le_gender.transform(["Female"])[0]],
    "Age": [45],
    "Tenure": [3],
    "Balance": [120000],
    "NumOfProducts": [1],
    "HasCrCard": [1],
    "IsActiveMember": [0],
    "EstimatedSalary": [90000]
})

prediction = best_model.predict(new_customer)[0]
probability = best_model.predict_proba(new_customer)[0][1]

print("\nLive Customer Prediction Example")
print("Prediction:", "Churned" if prediction == 1 else "Retained")
print(f"Churn Probability: {probability * 100:.2f}%")

if probability < 0.33:
    print("Risk Level: Low Risk")
elif probability < 0.66:
    print("Risk Level: Medium Risk")
else:
    print("Risk Level: High Risk")


# ============================================================
# 36. SAVE FINAL OUTPUT CSV
# ============================================================

bank_df.to_csv("bank_customer_retention_ml_output.csv", index=False)

print("\nFinal output saved as bank_customer_retention_ml_output.csv")


# ============================================================
# 37. SAVE ML MODEL AND ENCODERS
# ============================================================

import joblib

joblib.dump(best_model, "gradient_boosting_churn_model.pkl")
joblib.dump(le_geo, "geography_encoder.pkl")
joblib.dump(le_gender, "gender_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and encoders saved successfully.")


# ============================================================
# 38. FINAL PROJECT CONCLUSION
# ============================================================

print("=" * 70)
print("FINAL PROJECT CONCLUSION")
print("=" * 70)

print("""
This project analyzed customer churn in the banking sector using customer behavior,
product utilization, active membership, relationship strength, and machine learning.

Key Findings:
1. Customer churn is not only based on demographics.
2. Inactive customers show higher churn risk.
3. Product utilization strongly affects customer retention.
4. High-balance inactive customers are premium risk customers.
5. Relationship Strength Index helps identify weak customer relationships.
6. Gradient Boosting performed well for churn prediction.
7. ML churn probability helps banks identify risky customers in advance.

Final Outcome:
This project converts raw banking customer data into actionable retention intelligence.
The final Streamlit dashboard helps banks monitor churn, identify high-risk customers,
understand customer behavior, and take proactive retention actions.
""")
