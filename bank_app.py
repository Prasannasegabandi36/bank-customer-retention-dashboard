import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bank Customer Retention Dashboard",
    page_icon="🏦",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1f33 45%, #102a43 100%);
    color: white;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

.sub-title {
    font-size: 17px;
    color: #b8c7d9;
    margin-bottom: 25px;
}

.kpi-card {
    background: rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 8px 22px rgba(0,0,0,0.25);
}

.kpi-value {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
}

.kpi-label {
    font-size: 14px;
    color: #b8c7d9;
}

.alert-high {
    background: rgba(255, 75, 75, 0.18);
    border-left: 6px solid #ff4b4b;
    padding: 18px;
    border-radius: 12px;
    color: white;
}

.alert-medium {
    background: rgba(255, 193, 7, 0.18);
    border-left: 6px solid #ffc107;
    padding: 18px;
    border-radius: 12px;
    color: white;
}

.alert-good {
    background: rgba(0, 200, 120, 0.18);
    border-left: 6px solid #00c878;
    padding: 18px;
    border-radius: 12px;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank.csv")
    return df


df = load_data()


# ---------------------------------------------------------
# BASIC CLEANING
# ---------------------------------------------------------
df = df.dropna()

df["Customer_Status"] = df["Exited"].map({
    0: "Retained",
    1: "Churned"
})


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
df["Engagement_Segment"] = np.select(
    [
        (df["IsActiveMember"] == 1) & (df["NumOfProducts"] >= 2),
        (df["IsActiveMember"] == 0) & (df["Balance"] > 100000),
        (df["IsActiveMember"] == 1) & (df["NumOfProducts"] == 1),
        (df["IsActiveMember"] == 0)
    ],
    [
        "Active Multi-Product Customer",
        "Inactive High-Value Customer",
        "Active Single-Product Customer",
        "Inactive Low-Engagement Customer"
    ],
    default="Normal Customer"
)

df["Relationship_Score"] = (
    df["IsActiveMember"] * 25
    + np.where(df["NumOfProducts"] >= 2, 25, 10)
    + np.where(df["HasCrCard"] == 1, 15, 0)
    + np.where(df["Tenure"] >= 5, 20, 5)
    + np.where(df["Balance"] > 100000, 15, 5)
)

df["Relationship_Level"] = pd.cut(
    df["Relationship_Score"],
    bins=[0, 45, 75, 120],
    labels=["Weak", "Medium", "Strong"]
)


# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------
@st.cache_resource
def train_models(data):
    model_df = data.copy()

    geo_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()

    model_df["Geography_Encoded"] = geo_encoder.fit_transform(model_df["Geography"])
    model_df["Gender_Encoded"] = gender_encoder.fit_transform(model_df["Gender"])

    features = [
        "CreditScore",
        "Geography_Encoded",
        "Gender_Encoded",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary"
    ]

    X = model_df[features]
    y = model_df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    model_results = []

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = 0

        accuracy = accuracy_score(y_test, y_pred)

        model_results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "AUC Score": round(auc, 4)
        })

        trained_models[name] = model

    best_model = trained_models["Gradient Boosting"]

    best_pred = best_model.predict(X_test)
    best_prob = best_model.predict_proba(X_test)[:, 1]

    best_accuracy = accuracy_score(y_test, best_pred)
    best_auc = roc_auc_score(y_test, best_prob)

    report = classification_report(y_test, best_pred, output_dict=True)

    return (
        best_model,
        trained_models,
        pd.DataFrame(model_results),
        geo_encoder,
        gender_encoder,
        features,
        best_accuracy,
        best_auc,
        report,
        X_test,
        y_test
    )


(
    model,
    trained_models,
    model_results_df,
    geo_encoder,
    gender_encoder,
    features,
    best_accuracy,
    best_auc,
    report,
    X_test,
    y_test
) = train_models(df)


# ---------------------------------------------------------
# ML PREDICTION FOR EXISTING CUSTOMERS
# ---------------------------------------------------------
ml_df = df.copy()

ml_df["Geography_Encoded"] = geo_encoder.transform(ml_df["Geography"])
ml_df["Gender_Encoded"] = gender_encoder.transform(ml_df["Gender"])

df["Churn_Probability"] = model.predict_proba(ml_df[features])[:, 1]
df["Churn_Probability_Percent"] = (df["Churn_Probability"] * 100).round(2)

df["ML_Risk_Level"] = pd.cut(
    df["Churn_Probability"],
    bins=[0, 0.35, 0.65, 1],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    include_lowest=True
)


# ---------------------------------------------------------
# FEATURE IMPORTANCE FUNCTION
# ---------------------------------------------------------
def get_feature_importance(model, feature_names):
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return importance_df


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("🏦 Dashboard Controls")

geography_filter = st.sidebar.multiselect(
    "Select Geography",
    sorted(df["Geography"].unique()),
    default=sorted(df["Geography"].unique())
)

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    sorted(df["Gender"].unique()),
    default=sorted(df["Gender"].unique())
)

risk_filter = st.sidebar.multiselect(
    "Select Risk Level",
    ["Low Risk", "Medium Risk", "High Risk"],
    default=["Low Risk", "Medium Risk", "High Risk"]
)

active_filter = st.sidebar.multiselect(
    "Active Member Status",
    [0, 1],
    default=[0, 1],
    format_func=lambda x: "Active" if x == 1 else "Inactive"
)

age_range = st.sidebar.slider(
    "Select Age Range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max()))
)

filtered_df = df[
    (df["Geography"].isin(geography_filter))
    & (df["Gender"].isin(gender_filter))
    & (df["ML_Risk_Level"].astype(str).isin(risk_filter))
    & (df["IsActiveMember"].isin(active_filter))
    & (df["Age"].between(age_range[0], age_range[1]))
]


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    "<div class='main-title'>🏦 Bank Customer Retention & Churn Intelligence Dashboard</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Customer segmentation • churn prediction • feature importance • SHAP explainability • real-time risk prediction</div>",
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------
total_customers = len(filtered_df)
churned_customers = int(filtered_df["Exited"].sum())
retained_customers = total_customers - churned_customers

churn_rate = round((churned_customers / total_customers) * 100, 2) if total_customers > 0 else 0
retention_rate = round((retained_customers / total_customers) * 100, 2) if total_customers > 0 else 0

avg_balance = round(filtered_df["Balance"].mean(), 2) if total_customers > 0 else 0
avg_credit_score = round(filtered_df["CreditScore"].mean(), 2) if total_customers > 0 else 0

high_risk_count = len(filtered_df[filtered_df["ML_Risk_Level"].astype(str) == "High Risk"])

k1, k2, k3, k4, k5, k6 = st.columns(6)

kpi_values = [
    (k1, "Total Customers", f"{total_customers:,}"),
    (k2, "Churn Rate", f"{churn_rate}%"),
    (k3, "Retention Rate", f"{retention_rate}%"),
    (k4, "High Risk", f"{high_risk_count:,}"),
    (k5, "Avg Balance", f"€{avg_balance:,.0f}"),
    (k6, "Avg Credit Score", f"{avg_credit_score}")
]

for col, label, value in kpi_values:
    col.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-label'>{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# ---------------------------------------------------------
# ALERT BOX
# ---------------------------------------------------------
if churn_rate >= 25:
    st.markdown(
        f"""
        <div class='alert-high'>
        🚨 Critical Alert: Churn rate is {churn_rate}%. Immediate retention action is required.
        </div>
        """,
        unsafe_allow_html=True
    )
elif churn_rate >= 15:
    st.markdown(
        f"""
        <div class='alert-medium'>
        ⚠️ Warning: Churn rate is {churn_rate}%. Monitor inactive and high-risk customers.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <div class='alert-good'>
        ✅ Stable: Churn rate is {churn_rate}%. Customer retention is currently healthy.
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Dashboard Overview",
    "👥 Customer Segmentation",
    "🤖 Real-Time Prediction",
    "📌 Feature Importance",
    "🔍 SHAP Explainability",
    "📈 Business Insights",
    "📋 Data Explorer"
])


# ---------------------------------------------------------
# TAB 1: DASHBOARD OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.subheader("📊 Dashboard Overview")

    c1, c2 = st.columns(2)

    with c1:
        status_count = filtered_df["Customer_Status"].value_counts().reset_index()
        status_count.columns = ["Customer Status", "Count"]

        fig = px.pie(
            status_count,
            names="Customer Status",
            values="Count",
            hole=0.55,
            title="Customer Retention vs Churn"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        geo_churn = filtered_df.groupby("Geography")["Exited"].mean().reset_index()
        geo_churn["Churn Rate (%)"] = (geo_churn["Exited"] * 100).round(2)

        fig = px.bar(
            geo_churn,
            x="Geography",
            y="Churn Rate (%)",
            text="Churn Rate (%)",
            title="Churn Rate by Geography"
        )

        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        gender_churn = filtered_df.groupby("Gender")["Exited"].mean().reset_index()
        gender_churn["Churn Rate (%)"] = (gender_churn["Exited"] * 100).round(2)

        fig = px.bar(
            gender_churn,
            x="Gender",
            y="Churn Rate (%)",
            text="Churn Rate (%)",
            title="Churn Rate by Gender"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c4:
        risk_count = filtered_df["ML_Risk_Level"].value_counts().reset_index()
        risk_count.columns = ["Risk Level", "Customer Count"]

        fig = px.bar(
            risk_count,
            x="Risk Level",
            y="Customer Count",
            text="Customer Count",
            title="ML Risk Level Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)

    with c5:
        active_churn = filtered_df.groupby("IsActiveMember")["Exited"].mean().reset_index()
        active_churn["Member Status"] = active_churn["IsActiveMember"].map({
            0: "Inactive",
            1: "Active"
        })
        active_churn["Churn Rate (%)"] = (active_churn["Exited"] * 100).round(2)

        fig = px.bar(
            active_churn,
            x="Member Status",
            y="Churn Rate (%)",
            text="Churn Rate (%)",
            title="Active vs Inactive Member Churn"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c6:
        product_churn = filtered_df.groupby("NumOfProducts")["Exited"].mean().reset_index()
        product_churn["Churn Rate (%)"] = (product_churn["Exited"] * 100).round(2)

        fig = px.bar(
            product_churn,
            x="NumOfProducts",
            y="Churn Rate (%)",
            text="Churn Rate (%)",
            title="Number of Products vs Churn"
        )

        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# TAB 2: CUSTOMER SEGMENTATION
# ---------------------------------------------------------
with tab2:
    st.subheader("👥 Customer Segmentation Analysis")

    c1, c2 = st.columns(2)

    with c1:
        segment_count = filtered_df["Engagement_Segment"].value_counts().reset_index()
        segment_count.columns = ["Segment", "Customer Count"]

        fig = px.bar(
            segment_count,
            x="Customer Count",
            y="Segment",
            orientation="h",
            text="Customer Count",
            title="Customer Engagement Segments"
        )

        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        segment_churn = filtered_df.groupby("Engagement_Segment")["Exited"].mean().reset_index()
        segment_churn["Churn Rate (%)"] = (segment_churn["Exited"] * 100).round(2)

        fig = px.bar(
            segment_churn,
            x="Engagement_Segment",
            y="Churn Rate (%)",
            text="Churn Rate (%)",
            title="Churn Rate by Segment"
        )

        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Relationship Strength Summary")

    relationship_summary = filtered_df.groupby("Relationship_Level").agg(
        Customers=("CustomerId", "count"),
        Avg_Balance=("Balance", "mean"),
        Avg_CreditScore=("CreditScore", "mean"),
        Avg_Relationship_Score=("Relationship_Score", "mean"),
        Churn_Rate=("Exited", "mean")
    ).reset_index()

    relationship_summary["Avg_Balance"] = relationship_summary["Avg_Balance"].round(2)
    relationship_summary["Avg_CreditScore"] = relationship_summary["Avg_CreditScore"].round(2)
    relationship_summary["Avg_Relationship_Score"] = relationship_summary["Avg_Relationship_Score"].round(2)
    relationship_summary["Churn_Rate"] = (relationship_summary["Churn_Rate"] * 100).round(2)

    st.dataframe(relationship_summary, use_container_width=True)

    st.markdown("### High-Risk Customer Radar")

    fig = px.scatter(
        filtered_df,
        x="Age",
        y="Balance",
        color="ML_Risk_Level",
        size="Churn_Probability_Percent",
        hover_data=[
            "CustomerId",
            "Surname",
            "Geography",
            "Gender",
            "NumOfProducts",
            "IsActiveMember",
            "Churn_Probability_Percent"
        ],
        title="Customer Risk Radar: Age vs Balance"
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# TAB 3: REAL-TIME PREDICTION
# ---------------------------------------------------------
with tab3:
    st.subheader("🤖 Real-Time Customer Churn Prediction")

    st.write(
        "Enter customer details below to instantly predict churn probability and customer risk level."
    )

    with st.form("real_time_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=900,
                value=650
            )

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=40
            )

            tenure = st.number_input(
                "Tenure",
                min_value=0,
                max_value=10,
                value=5
            )

        with col2:
            balance = st.number_input(
                "Balance",
                min_value=0.0,
                max_value=300000.0,
                value=75000.0
            )

            num_products = st.selectbox(
                "Number of Products",
                [1, 2, 3, 4]
            )

            estimated_salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                max_value=250000.0,
                value=100000.0
            )

        with col3:
            geography = st.selectbox(
                "Geography",
                sorted(df["Geography"].unique())
            )

            gender = st.selectbox(
                "Gender",
                sorted(df["Gender"].unique())
            )

            has_card = st.selectbox(
                "Has Credit Card",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            is_active = st.selectbox(
                "Active Member",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Geography_Encoded": [geo_encoder.transform([geography])[0]],
            "Gender_Encoded": [gender_encoder.transform([gender])[0]],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_products],
            "HasCrCard": [has_card],
            "IsActiveMember": [is_active],
            "EstimatedSalary": [estimated_salary]
        })

        churn_probability = model.predict_proba(input_data)[0][1]
        churn_percentage = round(churn_probability * 100, 2)

        st.markdown("### Prediction Result")

        if churn_probability >= 0.65:
            st.error(f"🚨 High Churn Risk: {churn_percentage}%")
            st.write(
                "Recommended Action: Assign a relationship manager, offer loyalty benefits, "
                "and contact the customer immediately."
            )

        elif churn_probability >= 0.35:
            st.warning(f"⚠️ Medium Churn Risk: {churn_percentage}%")
            st.write(
                "Recommended Action: Send personalized offers and monitor customer engagement."
            )

        else:
            st.success(f"✅ Low Churn Risk: {churn_percentage}%")
            st.write(
                "Recommended Action: Continue normal engagement and cross-sell suitable products."
            )

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_percentage,
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 35], "color": "lightgreen"},
                    {"range": [35, 65], "color": "khaki"},
                    {"range": [65, 100], "color": "salmon"}
                ],
                "bar": {"color": "royalblue"}
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        st.markdown("### Input Customer Details")

        st.dataframe(input_data, use_container_width=True)


# ---------------------------------------------------------
# TAB 4: FEATURE IMPORTANCE
# ---------------------------------------------------------
with tab4:
    st.subheader("📌 Model Feature Importance")

    st.write(
        "Feature importance explains which customer attributes influence churn prediction the most."
    )

    importance_df = get_feature_importance(model, features)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title="Top Features Influencing Customer Churn"
    )

    fig.update_traces(texttemplate="%{text:.4f}")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(importance_df, use_container_width=True)

    st.markdown("""
    ### Business Interpretation

    This section helps the banking team understand why customers are likely to churn.

    - If **Age** is important, churn behavior changes strongly across customer age groups.
    - If **IsActiveMember** is important, inactive customers need retention campaigns.
    - If **NumOfProducts** is important, product usage affects customer loyalty.
    - If **Balance** is important, high-value customers should be monitored carefully.
    - If **Geography** is important, region-specific customer retention strategies are needed.
    """)


# ---------------------------------------------------------
# TAB 5: SHAP EXPLAINABILITY
# ---------------------------------------------------------
with tab5:
    st.subheader("🔍 SHAP Model Explainability")

    st.write(
        "SHAP analysis gives deeper explainability by showing how each feature pushes predictions toward churn or retention."
    )

    if SHAP_AVAILABLE:
        try:
            sample_data = ml_df[features].sample(
                min(500, len(ml_df)),
                random_state=42
            )

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_data)

            st.markdown("### SHAP Summary Plot")

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, sample_data, show=False)
            st.pyplot(fig)

            st.markdown("""
            ### How to Read This SHAP Chart

            - Features at the top have the highest impact on prediction.
            - Points moving right increase churn probability.
            - Points moving left reduce churn probability.
            - Red points usually represent high feature values.
            - Blue points usually represent low feature values.
            """)

        except Exception as e:
            st.warning("SHAP chart could not be generated in this environment.")
            st.write("Reason:", e)
            st.info(
                "Feature importance is still available as a model interpretability method."
            )

    else:
        st.warning("SHAP is not installed.")
        st.write(
            "Please add `shap` in requirements.txt to enable SHAP explainability."
        )


# ---------------------------------------------------------
# TAB 6: BUSINESS INSIGHTS
# ---------------------------------------------------------
with tab6:
    st.subheader("📈 Business Insights & Retention Strategy")

    st.markdown("""
    ## Key Business Insights

    ### 1. Inactive customers show higher churn risk
    Customers who are not active members are more likely to leave the bank.
    This means customer engagement is one of the most important retention factors.

    ### 2. Product utilization affects customer loyalty
    Customers using fewer banking products may have weaker relationships with the bank.
    Cross-selling relevant products can improve retention.

    ### 3. High-balance inactive customers are important
    Some customers may have high balances but low engagement.
    These customers are valuable and should be prioritized for retention campaigns.

    ### 4. Machine learning helps identify risk early
    Instead of waiting for customers to leave, the bank can use churn probability to identify high-risk customers early.

    ## Recommended Retention Actions

    - Contact high-risk customers through relationship managers.
    - Offer loyalty rewards to inactive high-value customers.
    - Create personalized offers for single-product customers.
    - Monitor churn probability regularly.
    - Use geography-based retention campaigns.
    - Use feature importance and SHAP analysis to understand churn reasons.
    """)

    st.markdown("### Model Comparison")

    st.dataframe(model_results_df, use_container_width=True)

    fig = px.bar(
        model_results_df,
        x="Model",
        y="AUC Score",
        text="AUC Score",
        title="Model Performance Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    ### Best Model Used

    The final model used in this dashboard is **Gradient Boosting Classifier**.

    - Accuracy: **{round(best_accuracy * 100, 2)}%**
    - AUC Score: **{round(best_auc, 4)}**

    Gradient Boosting was selected because it provides strong predictive performance and supports feature importance analysis.
    """)


# ---------------------------------------------------------
# TAB 7: DATA EXPLORER
# ---------------------------------------------------------
with tab7:
    st.subheader("📋 Data Explorer")

    display_columns = [
        "CustomerId",
        "Surname",
        "Geography",
        "Gender",
        "Age",
        "CreditScore",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "Tenure",
        "EstimatedSalary",
        "Exited",
        "Customer_Status",
        "Engagement_Segment",
        "Relationship_Score",
        "Relationship_Level",
        "Churn_Probability_Percent",
        "ML_Risk_Level"
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True
    )

    csv = filtered_df[display_columns].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Customer Data",
        data=csv,
        file_name="bank_customer_retention_filtered_data.csv",
        mime="text/csv"
    )

    st.markdown("### Top 25 High-Risk Customers")

    high_risk_customers = filtered_df.sort_values(
        by="Churn_Probability_Percent",
        ascending=False
    ).head(25)

    st.dataframe(
        high_risk_customers[display_columns],
        use_container_width=True
    )

    high_risk_csv = high_risk_customers[display_columns].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download High-Risk Customers",
        data=high_risk_csv,
        file_name="high_risk_bank_customers.csv",
        mime="text/csv"
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")

st.markdown("""
### Project Summary

This project is an end-to-end banking analytics dashboard for customer retention and churn prediction.
It combines data visualization, customer segmentation, machine learning, model explainability, and real-time prediction.

The dashboard helps banking teams:
- Understand customer churn behavior
- Identify high-risk customers
- Predict churn probability
- Explain model decisions using feature importance and SHAP
- Take early retention actions
""")
