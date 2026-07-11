
#PHASE 4 : EXPLORATORY DATA ANALYSIS (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")

# Create Output Folder


os.makedirs("outputs/graphs", exist_ok=True)


# Load Clean Dataset


df = pd.read_csv("outputs/cleaned_data/cleaned_bank_churn.csv")

print("="*80)
print("PHASE 4 : EXPLORATORY DATA ANALYSIS")
print("="*80)

print("\nDataset Loaded Successfully")

print("\nDataset Shape :", df.shape)


# Dataset Information

print("\nDataset Information")

print(df.info())

print("\nStatistical Summary")

print(df.describe())
# Target Variable Distribution

plt.figure(figsize=(6,5))

sns.countplot(data=df,x="Exited")

plt.title("Customer Churn Distribution")

plt.savefig("outputs/graphs/churn_distribution.png")

plt.show()
# Churn Percentage

plt.figure(figsize=(6,6))

df["Exited"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=["skyblue","orange"]
)

plt.ylabel("")

plt.title("Customer Churn Percentage")

plt.savefig("outputs/graphs/churn_percentage.png")

plt.show()
# AGE DISTRIBUTION

plt.figure(figsize=(8,5))

sns.histplot(df["Age"],bins=30,kde=True)

plt.title("Age Distribution")

plt.savefig("outputs/graphs/age_distribution.png")

plt.show()

# CREDIT SCORE DISTRIBUTION

plt.figure(figsize=(8,5))

sns.histplot(df["CreditScore"],bins=30,kde=True,color="green")

plt.title("Credit Score Distribution")

plt.savefig("outputs/graphs/credit_score_distribution.png")

plt.show()
# BALANCE DISTRIBUTION

plt.figure(figsize=(8,5))

sns.histplot(df["Balance"],bins=30,kde=True,color="purple")

plt.title("Balance Distribution")

plt.savefig("outputs/graphs/balance_distribution.png")

plt.show()
# SALARY DISTRIBUTION

plt.figure(figsize=(8,5))

sns.histplot(df["EstimatedSalary"],bins=30,kde=True,color="red")

plt.title("Estimated Salary Distribution")

plt.savefig("outputs/graphs/salary_distribution.png")

plt.show()
# TENURE DISTRIBUTION

plt.figure(figsize=(8,5))

sns.countplot(data=df,x="Tenure")

plt.title("Customer Tenure")

plt.savefig("outputs/graphs/tenure_distribution.png")

plt.show()


# NUMBER OF PRODUCTS


plt.figure(figsize=(8,5))

sns.countplot(data=df,x="NumOfProducts")

plt.title("Number of Products")

plt.savefig("outputs/graphs/products_distribution.png")

plt.show()
# GENDER DISTRIBUTION

plt.figure(figsize=(6,5))

sns.countplot(data=df,x="Gender")

plt.title("Gender Distribution")

plt.savefig("outputs/graphs/gender_distribution.png")

plt.show()
# GEOGRAPHY DISTRIBUTION

plt.figure(figsize=(8,5))

sns.countplot(data=df,x="Geography")

plt.title("Country Distribution")

plt.savefig("outputs/graphs/geography_distribution.png")

plt.show()
# ACTIVE MEMBERS

plt.figure(figsize=(6,5))

sns.countplot(data=df,x="IsActiveMember")

plt.title("Active Members")

plt.savefig("outputs/graphs/active_members.png")

plt.show()
# CREDIT CARD HOLDERS

plt.figure(figsize=(6,5))

sns.countplot(data=df,x="HasCrCard")

plt.title("Credit Card Holders")

plt.savefig("outputs/graphs/credit_card_holders.png")

plt.show()
# BOXPLOTS

numerical_columns = [

    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "EstimatedSalary"

]

for column in numerical_columns:

    plt.figure(figsize=(8,4))

    sns.boxplot(x=df[column])

    plt.title(f"Boxplot of {column}")

    plt.savefig(f"outputs/graphs/{column}_boxplot.png")

    plt.show()
# KDE PLOTS


for column in numerical_columns:

    plt.figure(figsize=(8,4))

    sns.kdeplot(df[column],fill=True)

    plt.title(f"KDE Plot : {column}")

    plt.savefig(f"outputs/graphs/{column}_kde.png")

    plt.show()


# SUMMARY


print("\nUNIVARIATE ANALYSIS COMPLETED")

print("Graphs Saved Successfully.")

print("="*80)
print("NEXT : BIVARIATE ANALYSIS")
print("="*80)



   # PHASE 4 : BIVARIATE ANALYSIS


print("\n" + "="*80)
print("BIVARIATE ANALYSIS")
print("="*80)


# AGE vs CHURN


plt.figure(figsize=(8,5))

sns.boxplot(x="Exited", y="Age", data=df)

plt.title("Age vs Customer Churn")

plt.savefig("outputs/graphs/age_vs_churn.png")

plt.show()

print("Insight: Customers with higher age tend to churn more frequently.\n")


# CREDIT SCORE vs CHURN


plt.figure(figsize=(8,5))

sns.boxplot(x="Exited", y="CreditScore", data=df)

plt.title("Credit Score vs Churn")

plt.savefig("outputs/graphs/creditscore_vs_churn.png")

plt.show()

print("Insight: Customers with lower credit scores generally show slightly higher churn.\n")


# BALANCE vs CHURN


plt.figure(figsize=(8,5))

sns.boxplot(x="Exited", y="Balance", data=df)

plt.title("Balance vs Churn")

plt.savefig("outputs/graphs/balance_vs_churn.png")

plt.show()

print("Insight: Customers with larger account balances appear more likely to churn.\n")


# ESTIMATED SALARY vs CHURN


plt.figure(figsize=(8,5))

sns.boxplot(x="Exited", y="EstimatedSalary", data=df)

plt.title("Estimated Salary vs Churn")

plt.savefig("outputs/graphs/salary_vs_churn.png")

plt.show()

print("Insight: Salary alone has a weak relationship with churn.\n")


# TENURE vs CHURN


plt.figure(figsize=(8,5))

sns.countplot(x="Tenure", hue="Exited", data=df)

plt.title("Tenure vs Churn")

plt.savefig("outputs/graphs/tenure_vs_churn.png")

plt.show()

print("Insight: Customers with shorter tenure generally churn more often.\n")


# GENDER vs CHURN


plt.figure(figsize=(6,5))

sns.countplot(x="Gender", hue="Exited", data=df)

plt.title("Gender vs Churn")

plt.savefig("outputs/graphs/gender_vs_churn.png")

plt.show()

print("Insight: Compare churn counts across genders.\n")


# GEOGRAPHY vs CHURN


plt.figure(figsize=(8,5))

sns.countplot(x="Geography", hue="Exited", data=df)

plt.title("Country vs Churn")

plt.savefig("outputs/graphs/geography_vs_churn.png")

plt.show()

print("Insight: Some countries show higher churn than others.\n")


# NUMBER OF PRODUCTS vs CHURN


plt.figure(figsize=(8,5))

sns.countplot(x="NumOfProducts", hue="Exited", data=df)

plt.title("Products vs Churn")

plt.savefig("outputs/graphs/products_vs_churn.png")

plt.show()

print("Insight: Customers with fewer products often churn more frequently.\n")


# ACTIVE MEMBER vs CHURN


plt.figure(figsize=(6,5))

sns.countplot(x="IsActiveMember", hue="Exited", data=df)

plt.title("Active Member vs Churn")

plt.savefig("outputs/graphs/active_vs_churn.png")

plt.show()

print("Insight: Inactive members exhibit a noticeably higher churn rate.\n")


# CREDIT CARD vs CHURN


plt.figure(figsize=(6,5))

sns.countplot(x="HasCrCard", hue="Exited", data=df)

plt.title("Credit Card Ownership vs Churn")

plt.savefig("outputs/graphs/creditcard_vs_churn.png")

plt.show()

print("Insight: Credit card ownership alone has limited influence on churn.\n")


# AGE DISTRIBUTION BY CHURN


plt.figure(figsize=(8,5))

sns.histplot(data=df,
    x="Age",
    hue="Exited",
    kde=True,
    bins=25)

plt.title("Age Distribution by Churn")

plt.savefig("outputs/graphs/age_distribution_churn.png")

plt.show()


# BALANCE DISTRIBUTION BY CHURN


plt.figure(figsize=(8,5))

sns.histplot(data=df,
    x="Balance",
    hue="Exited",
    bins=30)

plt.title("Balance Distribution by Churn")

plt.savefig("outputs/graphs/balance_distribution_churn.png")

plt.show()


# SCATTER PLOT : AGE vs BALANCE


plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Age",
    y="Balance",
    hue="Exited",
    alpha=0.7
)

plt.title("Age vs Balance (Churn)")

plt.savefig("outputs/graphs/age_balance_scatter.png")

plt.show()


# VIOLIN PLOT : AGE


plt.figure(figsize=(8,5))

sns.violinplot(x="Exited", y="Age", data=df)

plt.title("Age Distribution by Churn")

plt.savefig("outputs/graphs/age_violin.png")

plt.show()


# VIOLIN PLOT : BALANCE


plt.figure(figsize=(8,5))

sns.violinplot(x="Exited", y="Balance", data=df)

plt.title("Balance Distribution by Churn")

plt.savefig("outputs/graphs/balance_violin.png")

plt.show()


# CHURN RATE BY COUNTRY


country_churn = (
    df.groupby("Geography")["Exited"]
     .mean()
      .sort_values(ascending=False)
)

plt.figure(figsize=(8,5))

country_churn.plot(kind="bar")

plt.title("Average Churn Rate by Country")

plt.ylabel("Average Churn")

plt.tight_layout()

plt.savefig("outputs/graphs/churn_rate_country.png")

plt.show()

print("\nBIVARIATE ANALYSIS COMPLETED")

print("="*80)
print("NEXT : MULTIVARIATE ANALYSIS")
print("="*80)




#PHASE 4 : MULTIVARIATE ANALYSIS


print("\n" + "="*80)
print("MULTIVARIATE ANALYSIS")
print("="*80)


# Correlation Heatmap


print("\nGenerating Correlation Heatmap...")

plt.figure(figsize=(14,10))

correlation = df.corr(numeric_only=True)

sns.heatmap(correlation,
   annot=True,
   cmap="coolwarm",
   linewidths=0.5,
   fmt=".2f")

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("outputs/graphs/correlation_heatmap.png")

plt.show()


# Pairplot


print("\nGenerating Pairplot...")

pairplot_columns = [
    "CreditScore",
    "Age",
    "Balance",
    "EstimatedSalary",
    "Exited"
]

sns.pairplot(
    df[pairplot_columns],
    hue="Exited",
    diag_kind="hist"
)

plt.savefig("outputs/graphs/pairplot.png")

plt.show()


# Scatter Plot : Age vs Salary


plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Age",
    y="EstimatedSalary",
    hue="Exited"
)

plt.title("Age vs Estimated Salary")

plt.tight_layout()

plt.savefig("outputs/graphs/age_salary_scatter.png")

plt.show()


# Scatter Plot : Credit Score vs Balance


plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="CreditScore",
    y="Balance",
    hue="Exited"
)

plt.title("Credit Score vs Balance")

plt.tight_layout()

plt.savefig("outputs/graphs/credit_balance_scatter.png")

plt.show()


# Age vs Products


plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Age",
    y="NumOfProducts",
    hue="Exited"
)

plt.title("Age vs Number of Products")

plt.tight_layout()

plt.savefig("outputs/graphs/age_products_scatter.png")

plt.show()


# Balance vs Salary


plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Balance",
    y="EstimatedSalary",
    hue="Exited"
)

plt.title("Balance vs Estimated Salary")

plt.tight_layout()

plt.savefig("outputs/graphs/balance_salary_scatter.png")

plt.show()


# Churn Rate by Age Group


age_group = pd.cut(
    df["Age"],
    bins=[18,30,40,50,60,100],
    labels=["18-30","31-40","41-50","51-60","60+"]
)

age_churn = df.groupby(age_group)["Exited"].mean()

plt.figure(figsize=(8,5))

age_churn.plot(kind="bar")

plt.title("Average Churn Rate by Age Group")

plt.ylabel("Average Churn")

plt.tight_layout()

plt.savefig("outputs/graphs/churn_age_group.png")

plt.show()


# Churn Rate by Products


product_churn = df.groupby("NumOfProducts")["Exited"].mean()

plt.figure(figsize=(8,5))

product_churn.plot(kind="bar")

plt.title("Average Churn by Number of Products")

plt.ylabel("Average Churn")

plt.tight_layout()

plt.savefig("outputs/graphs/churn_products.png")

plt.show()


# Churn Rate by Active Member


active_churn = df.groupby("IsActiveMember")["Exited"].mean()

plt.figure(figsize=(6,5))

active_churn.plot(kind="bar")

plt.title("Average Churn by Active Membership")

plt.ylabel("Average Churn")

plt.tight_layout()

plt.savefig("outputs/graphs/churn_active_member.png")

plt.show()


# Churn Rate by Gender


gender_churn = df.groupby("Gender")["Exited"].mean()

plt.figure(figsize=(6,5))

gender_churn.plot(kind="bar")

plt.title("Average Churn by Gender")

plt.ylabel("Average Churn")

plt.tight_layout()

plt.savefig("outputs/graphs/churn_gender.png")

plt.show()


# Churn Rate by Credit Card


credit_churn = df.groupby("HasCrCard")["Exited"].mean()

plt.figure(figsize=(6,5))

credit_churn.plot(kind="bar")

plt.title("Average Churn by Credit Card Ownership")

plt.ylabel("Average Churn")

plt.tight_layout()

plt.savefig("outputs/graphs/churn_creditcard.png")

plt.show()


# Top Correlated Features


print("\nTop Correlations with Exited\n")

corr_target = correlation["Exited"].sort_values(ascending=False)

print(corr_target)


# Business Insights


print("\n" + "="*80)
print("BUSINESS INSIGHTS")
print("="*80)

print("""
1. Older customers generally exhibit a higher churn tendency.

2. Inactive members are significantly more likely to leave the bank.

3. Customers with only one product tend to churn more frequently.

4. Customers with higher balances show an increased likelihood of churn.

5. Geography influences churn behaviour, indicating regional differences.

6. Estimated Salary alone is not a strong predictor of churn.

7. Credit card ownership has a relatively small impact on churn.

8. Customer engagement appears to be one of the strongest indicators of retention.
""")


# Save EDA Summary


summary = pd.DataFrame({

    "Observation":[

        "Higher Age",

        "Inactive Members",

        "Single Product",

        "Higher Balance",

        "Geography",

        "Estimated Salary",

        "Credit Card"

    ],

    "Finding":[

        "Higher Churn",

        "Higher Churn",

        "Higher Churn",

        "Higher Churn",

        "Region Dependent",

        "Weak Effect",

        "Low Effect"

    ]

})

summary.to_csv(
    "outputs/reports/eda_summary.csv",
    index=False
)

print("\nEDA Summary Saved Successfully.")

print("\nGraphs Saved Successfully.")

print("\n" + "="*80)
print("PHASE 4 COMPLETED SUCCESSFULLY")
print("="*80)

print("\nNext Phase : FEATURE ENGINEERING")



#  PHASE 5 : FEATURE ENGINEERING


print("\n" + "="*80)
print("PHASE 5 : FEATURE ENGINEERING")
print("="*80)


# Create Copy


feature_df = df.copy()

print("\nCreating New Features...")


# Feature 1 : Balance-to-Salary Ratio


feature_df["BalanceSalaryRatio"] = (
    feature_df["Balance"] /
    (feature_df["EstimatedSalary"] + 1)
)

print("BalanceSalaryRatio Created")


# Feature 2 : Product Density


feature_df["ProductDensity"] = (
    feature_df["NumOfProducts"] /
    (feature_df["Tenure"] + 1)
)

print("ProductDensity Created")


# Feature 3 : Engagement Score


feature_df["EngagementScore"] = (
    feature_df["HasCrCard"] +
    feature_df["IsActiveMember"]
)

print("EngagementScore Created")


# Feature 4 : Age-Tenure Interaction


feature_df["AgeTenureInteraction"] = (
    feature_df["Age"] *
    feature_df["Tenure"]
)

print("AgeTenureInteraction Created")


# Feature 5 : Balance Per Product


feature_df["BalancePerProduct"] = (
    feature_df["Balance"] /
    (feature_df["NumOfProducts"] + 1)
)

print("BalancePerProduct Created")


# Feature 6 : Credit Score Category


def credit_category(score):

    if score < 580:
        return "Poor"

    elif score < 670:
        return "Fair"

    elif score < 740:
        return "Good"

    else:
        return "Excellent"

feature_df["CreditCategory"] = feature_df["CreditScore"].apply(credit_category)

print("CreditCategory Created")


# Feature 7 : Age Group


feature_df["AgeGroup"] = pd.cut(

    feature_df["Age"],

    bins=[18,30,40,50,60,100],

    labels=["18-30","31-40","41-50","51-60","60+"]

)

print("AgeGroup Created")


# Feature 8 : High Balance Flag


median_balance = feature_df["Balance"].median()

feature_df["HighBalanceFlag"] = np.where(

    feature_df["Balance"] > median_balance,

    1,

    0

)

print("HighBalanceFlag Created")


# Feature 9 : High Salary Flag


median_salary = feature_df["EstimatedSalary"].median()

feature_df["HighSalaryFlag"] = np.where(

    feature_df["EstimatedSalary"] > median_salary,

    1,

    0

)

print("HighSalaryFlag Created")


# Feature 10 : Loyal Customer Flag


feature_df["LoyalCustomer"] = np.where(

    feature_df["Tenure"] >= 5,

    1,

    0

)

print("LoyalCustomer Created")


# Feature 11 : Senior Citizen Flag


feature_df["SeniorCitizen"] = np.where(

    feature_df["Age"] >= 60,

    1,

    0

)

print("SeniorCitizen Created")


# Feature 12 : Financial Stability Score


feature_df["FinancialScore"] = (

    feature_df["CreditScore"]

    +

    feature_df["Balance"] / 1000

)

print("FinancialScore Created")


# Display New Features


print("\nNew Features Added\n")

new_columns = [

    "BalanceSalaryRatio",

    "ProductDensity",

    "EngagementScore",

    "AgeTenureInteraction",

    "BalancePerProduct",

    "CreditCategory",

    "AgeGroup",

    "HighBalanceFlag",

    "HighSalaryFlag",

    "LoyalCustomer",

    "SeniorCitizen",

    "FinancialScore"

]

print(feature_df[new_columns].head())


# Correlation of New Numerical Features


plt.figure(figsize=(12,8))

new_numeric = feature_df.select_dtypes(include=["int64","float64"])

sns.heatmap(

    new_numeric.corr(),

    cmap="coolwarm",

    annot=False

)

plt.title("Correlation Heatmap After Feature Engineering")

plt.tight_layout()

plt.savefig("outputs/graphs/feature_engineering_heatmap.png")

plt.show()

# Feature Distribution

engineered_numeric = [

    "BalanceSalaryRatio",

    "ProductDensity",

    "EngagementScore",

    "AgeTenureInteraction",

    "BalancePerProduct",

    "FinancialScore"

]

for column in engineered_numeric:

    plt.figure(figsize=(8,4))

    sns.histplot(

        feature_df[column],

        bins=30,

        kde=True

    )

    plt.title(column)

    plt.tight_layout()

    plt.savefig(f"outputs/graphs/{column}.png")

    plt.show()

# Dataset Shape

print("\nDataset Shape After Feature Engineering")

print(feature_df.shape)
# Save Engineered Dataset

import os

os.makedirs("outputs/engineered_data", exist_ok=True)

feature_df.to_csv(

    "outputs/engineered_data/engineered_bank_churn.csv",

    index=False

)

print("\nEngineered Dataset Saved Successfully.")
# Feature Engineering Report

feature_report = pd.DataFrame({

    "Feature": new_columns,

    "Description":[

        "Balance / Salary",

        "Products / Tenure",

        "Customer Engagement",

        "Age x Tenure",

        "Balance Per Product",

        "Credit Category",

        "Age Group",

        "High Balance",

        "High Salary",

        "Loyal Customer",

        "Senior Citizen",

        "Financial Stability"

    ]

})

feature_report.to_csv(

    "outputs/reports/feature_engineering_report.csv",

    index=False

)

print("\nFeature Engineering Report Saved Successfully.")

# COMPLETION MESSAGE

print("\n" + "="*80)

print("PHASE 5 COMPLETED SUCCESSFULLY")

print("="*80)

print("\nEngineered Dataset Ready For Machine Learning.")