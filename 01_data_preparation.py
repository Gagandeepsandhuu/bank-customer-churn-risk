
# Project Title:
# Predictive Modeling and Risk Scoring for Bank Customer Churn
# Description:
# This project predicts customer churn using Machine Learning techniques,
# generates customer churn probability, and assigns churn risk scores
# to support proactive customer retention strategies

# Import Required Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import os

# Display Setting

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", None)

plt.style.use("ggplot")

print("="*80)
print("BANK CUSTOMER CHURN ANALYSIS")
print("="*80)

print("\nProject Started Successfully...\n")

# Create Output Directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/cleaned_data", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/graphs", exist_ok=True)

print("Output folders created successfully.\n")
# Load Dataset

dataset_path = r"C:\Users\91771\Desktop\bank churns.py\European_Bank.csv"

try:

    df = pd.read_csv(r"C:\Users\91771\Desktop\bank churns.py\European_Bank.csv")

    print("Dataset Loaded Successfully.\n")

except Exception as e:

    print("Dataset Loading Failed.")

    print(e)

    exit()

# Project Information


print("="*80)
print("PROJECT INFORMATION")
print("="*80)

print(f"Project Name : Predictive Modeling and Risk Scoring for Bank Customer Churn")

print(f"Organization : European Central Bank")

print(f"Date : {datetime.now().strftime('%d-%m-%Y')}")

print(f"Python Version : Ready")

print()
# Business Background

print("="*80)
print("BACKGROUND")
print("="*80)

background = """

Customer churn directly impacts:

• Customer Lifetime Value (CLV)

• Revenue Stability

• Cross-selling Opportunities

• Long-term Competitiveness

Traditional churn analysis explains
why customers left.

Our goal is to identify customers
BEFORE they leave the bank.

"""

print(background)
# Problem Statement

print("="*80)
print("PROBLEM STATEMENT")
print("="*80)

problem = """

Banks possess rich customer-level data
but often lack:

1. Accurate Churn Prediction

2. Customer Risk Scores

3. Explainable Machine Learning

As a result,

Retention campaigns become

Reactive instead of Proactive.

"""

print(problem)

# Primary Objectives
print("="*80)
print("PRIMARY OBJECTIVES")
print("="*80)

primary = [

"Predict Customer Churn",

"Generate Churn Probability",

"Generate Churn Risk Score",

"Identify Churn Drivers"

]

for i,obj in enumerate(primary,1):

    print(f"{i}. {obj}")

print()


# Secondary Objectives

print("="*80)
print("SECONDARY OBJECTIVES")
print("="*80)

secondary = [

"Reduce False Positives",

"Improve Model Interpretability",

"Support Scenario-Based Analysis"

]

for i,obj in enumerate(secondary,1):

    print(f"{i}. {obj}")

print()
# Machine Learning Workflow

print("="*80)
print("PROJECT WORKFLOW")
print("="*80)

workflow = [

"Dataset Understanding",

"Data Cleaning",

"Exploratory Data Analysis",

"Feature Engineering",

"Data Preprocessing",

"Model Development",

"Hyperparameter Tuning",

"Model Evaluation",

"Explainable AI",

"Risk Scoring",

"What-if Scenario Analysis",

"Streamlit Dashboard",

"Deployment"

]

for i,step in enumerate(workflow,1):

    print(f"{i}. {step}")

print()
# Dataset Overview

print("="*80)
print("DATASET OVERVIEW")
print("="*80)

print("Dataset Shape")

print(df.shape)

rows, cols = df.shape

print(f"\nTotal Rows    : {rows}")

print(f"Total Columns : {cols}")

print()
# Dataset Columns

print("="*80)
print("DATASET COLUMNS")
print("="*80)

for i,column in enumerate(df.columns,1):

    print(f"{i}. {column}")

print()

# First Five Records


print("="*80)
print("FIRST FIVE RECORDS")
print("="*80)

print(df.head())

print()
# Last Five Records

print("="*80)
print("LAST FIVE RECORDS")
print("="*80)

print(df.tail())

print()
# Target Variable

print("="*80)
print("TARGET VARIABLE")
print("="*80)

print("Target Column : Exited\n")

print("0 ---> Customer Retained")

print("1 ---> Customer Churned")

print()
# End of Part 1

print("="*80)
print("PART 1 COMPLETED SUCCESSFULLY")
print("="*80)

print("\nNext : Dataset Understanding (Part 2)")
          #  PHASE 2 : DATASET UNDERSTANDING

print("\n" + "="*80)
print("PHASE 2 : DATASET UNDERSTANDING")
print("="*80)

# Dataset Information


print("\n1. DATASET INFORMATION")
print("-"*80)

df.info()
# Data Types

print("\n2. DATA TYPES")
print("-"*80)

print(df.dtypes)
# Numerical Columns

print("\n3. NUMERICAL COLUMNS")
print("-"*80)

numerical_columns = df.select_dtypes(include=['int64','float64']).columns.tolist()

print(numerical_columns)

print(f"\nTotal Numerical Columns : {len(numerical_columns)}")
# Categorical Columns

print("\n4. CATEGORICAL COLUMNS")
print("-"*80)

categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

print(categorical_columns)

print(f"\nTotal Categorical Columns : {len(categorical_columns)}")

# Statistical Summary

print("\n5. STATISTICAL SUMMARY (NUMERICAL)")
print("-"*80)

print(df.describe())
# Statistical Summary (All Columns)

print("\n6. COMPLETE DATA SUMMARY")
print("-"*80)

print(df.describe(include='all'))
# Missing Values

print("\n7. MISSING VALUES")
print("-"*80)

missing_values = df.isnull().sum()

print(missing_values)

# Missing Percentage

missing_percentage = (df.isnull().sum()/len(df))*100

print("\nMissing Percentage")

print(missing_percentage.round(2))
# Visualize Missing Values

plt.figure(figsize=(12,5))

sns.heatmap(df.isnull(),
            cbar=False,
            cmap="viridis")

plt.title("Missing Value Heatmap")

plt.tight_layout()

plt.savefig("outputs/graphs/missing_values_heatmap.png")

plt.show()
# Duplicate Records

print("\n8. DUPLICATE RECORDS")
print("-"*80)

duplicates = df.duplicated().sum()

print(f"Duplicate Rows : {duplicates}")
# Unique Values
print("\n9. UNIQUE VALUES")
print("-"*80)

unique_df = pd.DataFrame({

    "Column":df.columns,

    "Unique Values":[df[col].nunique() for col in df.columns]

})

print(unique_df)
# Display Unique Categories

print("\n10. UNIQUE CATEGORIES")
print("-"*80)

for col in categorical_columns:

    print(f"\n{col}")

    print(df[col].unique())
# Target Variable Distribution

print("\n11. TARGET VARIABLE DISTRIBUTION")
print("-"*80)

print(df["Exited"].value_counts())

print("\nPercentage")

print(round(df["Exited"].value_counts(normalize=True)*100,2))
# Count Plot
plt.figure(figsize=(6,5))

sns.countplot(data=df,
              x="Exited",
              palette="Set2")

plt.title("Customer Churn Distribution")

plt.xlabel("Exited")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig("outputs/graphs/churn_countplot.png")

plt.show()
# Pie Chart

plt.figure(figsize=(6,6))

df["Exited"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90,
    explode=(0,0.08),
    shadow=True
)

plt.title("Customer Churn Percentage")

plt.ylabel("")

plt.tight_layout()

plt.savefig("outputs/graphs/churn_piechart.png")

plt.show()
# Correlation Matrix

print("\n12. CORRELATION MATRIX")
print("-"*80)

correlation = df.corr(numeric_only=True)

print(correlation)

plt.figure(figsize=(14,10))

sns.heatmap(correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("outputs/graphs/correlation_heatmap.png")

plt.show()
# Memory Usage

print("\n13. MEMORY USAGE")
print("-"*80)

memory = df.memory_usage(deep=True).sum() / 1024**2

print(f"Dataset Memory Usage : {memory:.2f} MB")
# Data Quality Report

print("\n14. DATA QUALITY REPORT")
print("-"*80)

print(f"Rows                 : {df.shape[0]}")
print(f"Columns              : {df.shape[1]}")
print(f"Missing Values       : {df.isnull().sum().sum()}")
print(f"Duplicate Rows       : {df.duplicated().sum()}")
print(f"Numerical Columns    : {len(numerical_columns)}")
print(f"Categorical Columns  : {len(categorical_columns)}")


# Save Data Quality Report

report = pd.DataFrame({

    "Metric":[
        "Rows",
        "Columns",
        "Missing Values",
        "Duplicate Rows",
        "Numerical Columns",
        "Categorical Columns"
    ],

    "Value":[
        df.shape[0],
        df.shape[1],
        df.isnull().sum().sum(),
        df.duplicated().sum(),
        len(numerical_columns),
        len(categorical_columns)
    ]

})

report.to_csv("outputs/reports/data_quality_report.csv",
              index=False)

print("\nData Quality Report Saved Successfully.")

print("\n" + "="*80)
print("PHASE 2 COMPLETED SUCCESSFULLY")
print("="*80)

print("\nNext Phase : DATA CLEANING")
# PHASE 3 : DATA CLEANING

print("="*80)
print("PHASE 3 : DATA CLEANING")
print("="*80)

# Create a copy of the original dataset
clean_df = df.copy()

print("\nOriginal Dataset Shape :", clean_df.shape)
# STEP 1 : Remove Unnecessary Columns

print("\nRemoving unnecessary columns...")

columns_to_remove = ["CustomerId", "Surname"]

existing_columns = [col for col in columns_to_remove if col in clean_df.columns]

clean_df.drop(columns=existing_columns, inplace=True)

print("Removed Columns :", existing_columns)

print("Current Shape :", clean_df.shape)
# STEP 2 : Check Missing Value

print("\nChecking Missing Values...\n")

missing_values = clean_df.isnull().sum()

missing_df = pd.DataFrame({
    "Column": missing_values.index,
    "Missing Values": missing_values.values,
    "Percentage": (missing_values.values/len(clean_df))*100
})

print(missing_df)
# STEP 3 : Handle Missing Values

print("\nHandling Missing Values...")

for column in clean_df.columns:

    if clean_df[column].isnull().sum() > 0:

        # Numeric Columns
        if pd.api.types.is_numeric_dtype(clean_df[column]):

            median_value = clean_df[column].median()

            clean_df[column] = clean_df[column].fillna(median_value)

        # Categorical Columns
        else:

            mode_value = clean_df[column].mode()[0]

            clean_df[column] = clean_df[column].fillna(mode_value)

print("Missing Values Handled Successfully.")

# Verify

print("\nRemaining Missing Values")

print(clean_df.isnull().sum())
# STEP 4 : Remove Duplicate Rows

print("\nChecking Duplicate Rows...")

duplicates_before = clean_df.duplicated().sum()

print("Duplicate Rows :", duplicates_before)

clean_df.drop_duplicates(inplace=True)

duplicates_after = clean_df.duplicated().sum()

print("Remaining Duplicate Rows :", duplicates_after)

print("Dataset Shape :", clean_df.shape)
# STEP 5 : Correct Data Types

print("\nChecking Data Types")

print(clean_df.dtypes)

# Convert integer columns if required

integer_columns = [

    "CreditScore",
    "Age",
    "Tenure",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "Exited"

]

for col in integer_columns:

    if col in clean_df.columns:

        clean_df[col] = clean_df[col].astype(int)

print("\nData Types Corrected.")
# STEP 6 : Detect Outliers

print("\nDetecting Outliers (IQR Method)")

numeric_columns = clean_df.select_dtypes(include=["int64","float64"]).columns

outlier_summary = []

for column in numeric_columns:

    if column == "Exited":
        continue

    Q1 = clean_df[column].quantile(0.25)

    Q3 = clean_df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5*IQR

    upper = Q3 + 1.5*IQR

    outliers = clean_df[
        (clean_df[column] < lower) |
        (clean_df[column] > upper)
    ]

    outlier_summary.append([column, len(outliers)])

outlier_report = pd.DataFrame(
    outlier_summary,
    columns=["Feature","Outliers"]
)

print(outlier_report)

# STEP 7 : Visualize Outliers

print("\nGenerating Boxplots...")

for column in numeric_columns:

    if column == "Exited":
        continue

    plt.figure(figsize=(8,4))

    sns.boxplot(x=clean_df[column])

    plt.title(f"Boxplot : {column}")

    plt.tight_layout()

    plt.show()
# STEP 8 : Check Class Distribution

print("\nTarget Variable Distribution")

print(clean_df["Exited"].value_counts())

print("\nPercentage")

print(round(clean_df["Exited"].value_counts(normalize=True)*100,2))
# STEP 9 : Final Dataset Validation

print("\nFinal Dataset Validation")

print("Rows :", clean_df.shape[0])

print("Columns :", clean_df.shape[1])

print("Missing Values :", clean_df.isnull().sum().sum())

print("Duplicate Rows :", clean_df.duplicated().sum())
# STEP 10 : Save Clean Dataset

import os

os.makedirs("outputs/cleaned_data", exist_ok=True)

clean_df.to_csv(
    "outputs/cleaned_data/cleaned_bank_churn.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully.")
# STEP 11 : Save Data Cleaning Report

report = pd.DataFrame({

    "Metric":[

        "Rows",

        "Columns",

        "Missing Values",

        "Duplicate Rows"

    ],

    "Value":[

        clean_df.shape[0],

        clean_df.shape[1],

        clean_df.isnull().sum().sum(),

        clean_df.duplicated().sum()

    ]

})

os.makedirs("outputs/reports", exist_ok=True)

report.to_csv(
    "outputs/reports/data_cleaning_report.csv",
    index=False
)

print("\nData Cleaning Report Saved Successfully.")

print("\n" + "="*80)
print("PHASE 3 : DATA CLEANING COMPLETED SUCCESSFULLY")
print("="*80)

print("\nClean Dataset is ready for Feature Engineering and EDA.")

