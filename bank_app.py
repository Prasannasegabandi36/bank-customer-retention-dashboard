import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(
    page_title="Bank Customer Retention Intelligence",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0d1b2a 45%, #102a43 100%);
    color: white;
}
.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0px;
}
.sub-title {
    font-size: 18px;
    color: #b8c7d9;
    margin-bottom: 25px;
}
.kpi-card {
    background: rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}
.kpi-label {
    color: #b8c7d9;
    font-size: 14px;
}
.kpi-value {
    color: #ffffff;
    font-size: 32px;
    font-weight: 800;
}
.insight-box {
    background: rgba(14, 216, 180, 0.12);
    padding: 18px;
    border-left: 5px solid #0ed8b4;
    border-radius: 12px;
    margin-top: 15px;
}
.warning-box {
    background: rgba(255, 181, 71, 0.12);
    padding: 18px;
    border-left: 5px solid #ffb547;
    border-radius: 12px;
    margin-top: 15px;
}
.danger-box {
    background: rgba(255, 75, 75, 0.12);
    padding: 18px;
    border-left: 5px solid #ff4b4b;
    border-radius: 12px;
    margin-top: 15px;
}
[data-testid="stSidebar"] {
    background: #06101f;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv")

df = load_data()

df["Engagement_Profile"] = np.select(
    [
        (df["IsActiveMember"] == 1) & (df["NumOfProducts"] >= 2),
        (df["IsActiveMember"] == 0) & (df["Balance"] > 100000),
        (df["IsActiveMember"] == 1) & (df["NumOfProducts"] == 1),
        (df["IsActiveMember"] == 0)
    ],
    [
        "Active Engaged",
        "Inactive High-Balance",
        "Active Low-Product",
        "Inactive Disengaged"
    ],
    default="Other"
)

df["Relationship_Score"] = (
    df["IsActiveMember"] +
    (df["NumOfProducts"] >= 2).astype(int) +
    df["HasCrCard"] +
    (df["Tenure"] >= 5).astype(int)
)

df["Relationship_Level"] = pd.cut(
    df["Relationship_Score"],
    bins=[-1, 1, 2, 4],
    labels=["Weak", "Medium", "Strong"]
)

@st.cache_resource
def train_model(data):
    model_df = data.copy()

    geo_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()

    model_df["Geography_enc"] = geo_encoder.fit_transform(model_df["Geography"])
    model_df["Gender_enc"] = gender_encoder.fit_transform(model_df["Gender"])

    features = [
        "CreditScore", "Geography_enc", "Gender_enc", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary"
    ]

    X = model_df[features]
    y = model_df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    return model, geo_encoder, gender_encoder, features, accuracy, auc

model, geo_encoder, gender_encoder, features, accuracy, auc = train_model(df)

df_model = df.copy()
df_model["Geography_enc"] = geo_encoder.transform(df_model["Geography"])
df_model["Gender_enc"] = gender_encoder.transform(df_model["Gender"])
df["Churn_Probability"] = model.predict_proba(df_model[features])[:, 1]

df["ML_Risk_Level"] = pd.cut(
    df["Churn_Probability"],
    bins=[0, 0.33, 0.66, 1],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

st.sidebar.title("🏦 Bank Retention")
st.sidebar.caption("Client Intelligence Dashboard")

geography = st.sidebar.multiselect(
    "Select Geography",
    options=df["Geography"].unique(),
    default=df["Geography"].unique()
)

gender = st.sidebar.multiselect(
    "Select Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

active_status = st.sidebar.multiselect(
    "Active Member",
    options=df["IsActiveMember"].unique(),
    default=df["IsActiveMember"].unique()
)

risk_filter = st.sidebar.multiselect(
    "ML Risk Level",
    options=df["ML_Risk_Level"].dropna().unique(),
    default=df["ML_Risk_Level"].dropna().unique()
)

filtered_df = df[
    (df["Geography"].isin(geography)) &
    (df["Gender"].isin(gender)) &
    (df["IsActiveMember"].isin(active_status)) &
    (df["ML_Risk_Level"].isin(risk_filter))
]

st.markdown('<h1 class="main-title">🏦 Customer Retention Intelligence Platform</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Behavioral Analytics · Product Utilization · Churn Risk Prediction · Banking Retention Strategy</p>',
    unsafe_allow_html=True
)

total_customers = len(filtered_df)
churned = int(filtered_df["Exited"].sum())
retained = total_customers - churned
churn_rate = round((churned / total_customers) * 100, 2) if total_customers > 0 else 0
avg_prob = round(filtered_df["Churn_Probability"].mean() * 100, 2) if total_customers > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Customers</div>
        <div class="kpi-value">{total_customers:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Retained Customers</div>
        <div class="kpi-value">{retained:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Churned Customers</div>
        <div class="kpi-value">{churned:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Churn Rate</div>
        <div class="kpi-value">{churn_rate}%</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg ML Risk</div>
        <div class="kpi-value">{avg_prob}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Overview",
    "🧠 Customer Behavior",
    "⚠️ High Risk Customers",
    "🤖 ML Predictor",
    "🎯 Retention Strategy"
])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        churn_data = filtered_df["Exited"].map({0: "Retained", 1: "Churned"}).value_counts().reset_index()
        churn_data.columns = ["Status", "Count"]

        fig = px.pie(
            churn_data,
            names="Status",
            values="Count",
            hole=0.55,
            color="Status",
            color_discrete_map={"Retained": "#0ed8b4", "Churned": "#ff4b4b"},
            title="Customer Churn Distribution"
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        geo_churn = filtered_df.groupby("Geography")["Exited"].mean().reset_index()
        geo_churn["Churn Rate"] = geo_churn["Exited"] * 100

        fig = px.bar(
            geo_churn,
            x="Geography",
            y="Churn Rate",
            text=geo_churn["Churn Rate"].round(1).astype(str) + "%",
            color="Churn Rate",
            color_continuous_scale=["#0ed8b4", "#ffb547", "#ff4b4b"],
            title="Churn Rate by Geography"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>Business Insight:</b> This dashboard helps identify where churn is happening, which customer groups are risky,
    and how relationship strength affects retention.
    </div>
    """, unsafe_allow_html=True)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        active_churn = filtered_df.groupby("IsActiveMember")["Exited"].mean().reset_index()
        active_churn["Status"] = active_churn["IsActiveMember"].map({0: "Inactive", 1: "Active"})
        active_churn["Churn Rate"] = active_churn["Exited"] * 100

        fig = px.bar(
            active_churn,
            x="Status",
            y="Churn Rate",
            text=active_churn["Churn Rate"].round(1).astype(str) + "%",
            color="Status",
            color_discrete_map={"Inactive": "#ff4b4b", "Active": "#0ed8b4"},
            title="Active vs Inactive Customer Churn"
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        product_churn = filtered_df.groupby("NumOfProducts")["Exited"].mean().reset_index()
        product_churn["Churn Rate"] = product_churn["Exited"] * 100

        fig = px.bar(
            product_churn,
            x="NumOfProducts",
            y="Churn Rate",
            text=product_churn["Churn Rate"].round(1).astype(str) + "%",
            color="Churn Rate",
            color_continuous_scale=["#0ed8b4", "#ffb547", "#ff4b4b"],
            title="Product Usage vs Churn"
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    profile_churn = filtered_df.groupby("Engagement_Profile")["Exited"].mean().reset_index()
    profile_churn["Churn Rate"] = profile_churn["Exited"] * 100

    fig = px.bar(
        profile_churn,
        x="Churn Rate",
        y="Engagement_Profile",
        orientation="h",
        text=profile_churn["Churn Rate"].round(1).astype(str) + "%",
        color="Churn Rate",
        color_continuous_scale=["#0ed8b4", "#ffb547", "#ff4b4b"],
        title="Churn by Engagement Profile"
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    high_risk = filtered_df[filtered_df["ML_Risk_Level"] == "High Risk"].sort_values(
        "Churn_Probability", ascending=False
    )

    st.markdown(f"""
    <div class="danger-box">
    <b>High Risk Customers:</b> {len(high_risk):,} customers are predicted as high churn risk by the ML model.
    These customers should be prioritized for retention campaigns.
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        high_risk[
            [
                "CustomerId", "Surname", "Geography", "Gender", "Age",
                "Balance", "NumOfProducts", "IsActiveMember",
                "Churn_Probability", "ML_Risk_Level"
            ]
        ].head(50),
        use_container_width=True
    )

with tab4:
    st.subheader("🤖 Live Customer Churn Prediction")

    st.markdown(f"""
    <div class="insight-box">
    <b>Model Used:</b> Gradient Boosting Classifier<br>
    <b>Accuracy:</b> {round(accuracy * 100, 2)}%<br>
    <b>AUC Score:</b> {round(auc, 3)}
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            credit_score = st.number_input("Credit Score", 300, 900, 650)
            age = st.number_input("Age", 18, 100, 40)
            tenure = st.number_input("Tenure", 0, 10, 5)

        with col2:
            balance = st.number_input("Balance", 0.0, 300000.0, 75000.0)
            products = st.selectbox("Number of Products", [1, 2, 3, 4])
            salary = st.number_input("Estimated Salary", 0.0, 250000.0, 100000.0)

        with col3:
            geography_input = st.selectbox("Geography", df["Geography"].unique())
            gender_input = st.selectbox("Gender", df["Gender"].unique())
            card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            active = st.selectbox("Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

        submit = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submit:
        input_df = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography_enc": geo_encoder.transform([geography_input])[0],
            "Gender_enc": gender_encoder.transform([gender_input])[0],
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": card,
            "IsActiveMember": active,
            "EstimatedSalary": salary
        }])

        probability = model.predict_proba(input_df)[0][1]
        percentage = round(probability * 100, 2)

        if probability >= 0.66:
            st.error(f"🚨 High Churn Risk: {percentage}%")
            st.write("Recommended action: Immediate relationship manager outreach and personalized retention offer.")
        elif probability >= 0.33:
            st.warning(f"⚠️ Medium Churn Risk: {percentage}%")
            st.write("Recommended action: Monitor customer engagement and provide targeted product benefits.")
        else:
            st.success(f"✅ Low Churn Risk: {percentage}%")
            st.write("Recommended action: Continue loyalty engagement and cross-sell suitable products.")

with tab5:
    st.markdown("""
    <div class="insight-box">
    <h3>🎯 Recommended Retention Strategy</h3>
    <b>1. Reactivate inactive customers:</b> Send personalized offers and service calls.<br><br>
    <b>2. Focus on single-product customers:</b> Cross-sell savings, credit card, or investment products.<br><br>
    <b>3. Protect premium customers:</b> High-balance inactive customers need priority attention.<br><br>
    <b>4. Use ML scoring monthly:</b> Score all customers and create a high-risk retention list.<br><br>
    <b>5. Improve relationship strength:</b> Increase active membership, tenure loyalty, and product usage.
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        filtered_df[
            [
                "CustomerId", "Surname", "Geography", "Gender", "Age",
                "CreditScore", "Balance", "NumOfProducts", "IsActiveMember",
                "Exited", "Churn_Probability", "ML_Risk_Level",
                "Engagement_Profile", "Relationship_Level"
            ]
        ],
        use_container_width=True
    )
