# ==========================================
# BANK CUSTOMER CHURN PREDICTION SYSTEM
# Streamlit Dashboard
# ==========================================

import streamlit as st
import pandas as pd
import joblib


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)


# ==========================================
# Load Model, Scaler & Features
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "outputs/models/best_model.pkl"
    )

    scaler = joblib.load(
        "outputs/models/scaler.pkl"
    )

    return model, scaler


@st.cache_data
def load_features():

    features = pd.read_csv(
        "outputs/cleaned_data/final_features.csv"
    )

    return features



model, scaler = load_model()

features = load_features()

feature_names = features.columns.tolist()



# ==========================================
# Title
# ==========================================

st.title(
    "🏦 Bank Customer Churn Prediction Dashboard"
)


st.markdown(
"""
### Predict customer churn risk using Machine Learning

This system predicts whether a bank customer is likely to leave
and provides a risk score for customer retention strategies.
"""
)



# ==========================================
# Sidebar Customer Input
# ==========================================

st.sidebar.header(
    "👤 Customer Details"
)


input_data = {}


for feature in feature_names:

    value = st.sidebar.number_input(
        label=feature,
        value=float(features[feature].mean())
    )

    input_data[feature] = value



input_df = pd.DataFrame(
    [input_data]
)



# ==========================================
# Prediction
# ==========================================


if st.sidebar.button(
    "🔍 Predict Customer"
):


    # Scaling input

    input_scaled = scaler.transform(
        input_df
    )


    # Prediction

    prediction = model.predict(
        input_scaled
    )[0]


    probability = model.predict_proba(
        input_scaled
    )[0][1]


    percentage = probability * 100



    # Risk Category

    if probability < 0.30:

        risk = "🟢 LOW RISK"


    elif probability < 0.70:

        risk = "🟡 MEDIUM RISK"


    else:

        risk = "🔴 HIGH RISK"



    # ======================================
    # Result Display
    # ======================================

    st.header(
        "📊 Prediction Result"
    )


    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to CHURN"
        )

    else:

        st.success(
            "✅ Customer is likely to STAY"
        )



    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Churn Probability",
            f"{percentage:.2f}%"
        )


    with col2:

        st.metric(
            "Risk Level",
            risk
        )



    # Probability Bar

    st.subheader(
        "Churn Probability Score"
    )


    st.progress(
        float(probability)
    )



    # Customer Details

    st.subheader(
        "Customer Information"
    )


    st.dataframe(
        input_df
    )



    # ======================================
    # Download Report
    # ======================================


    result = input_df.copy()


    result["Prediction"] = (

        "Churn"
        if prediction == 1
        else
        "Stay"

    )


    result["Probability"] = round(
        percentage,
        2
    )


    result["Risk"] = risk



    csv = result.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


    st.download_button(

        label="⬇️ Download Prediction Report",

        data=csv,

        file_name="customer_churn_prediction.csv",

        mime="text/csv"

    )




# ==========================================
# Feature Importance
# ==========================================


st.header(
    "📌 Feature Importance"
)


if hasattr(
    model,
    "feature_importances_"
):


    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })


    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )


    st.bar_chart(

        importance.set_index(
            "Feature"
        )

    )


else:


    st.info(

        "Feature importance is available only for tree-based models."

    )




# ==========================================
# Footer
# ==========================================

st.markdown("---")


st.markdown(
"""
Developed by **Gagandeep Kaur**

🏦 Machine Learning Based Bank Customer Churn Risk Prediction System
"""
)