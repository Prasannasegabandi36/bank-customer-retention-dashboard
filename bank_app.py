import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Real-Time Bank Customer Retention Dashboard",
    page_icon="🏦",
    layout="wide"
)


# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1f33 45%, #102a43 100%);
    color: white;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 0px;
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
    font-size: 34px;
    font-weight: 800;
    color: #ffffff;
}
.kpi-label {
    font-size: 14px;
    color: #b8c7d9;
}
.alert-high {
    background: rgba(255, 75, 75, 0.15);
    border-left: 6px solid #ff4b4b;
    padding: 18px;
    border-radius: 12px;
    color: white;
}
.alert-medium {
    background: rgba(255, 193, 7, 0.15);
    border-left: 6px solid #ffc107;
    padding: 18px;
    border-radius: 12px;
    color: white;
}
.alert-good {
    background: rgba(0, 200, 120, 0.15);
    border-left: 6px solid #00c878;
    padding: 18px;
    border-radius: 12px;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank.csv")
    return df

df = load_data()


# ---------------- FEATURE ENGINEERING ----------------
df["Customer_Status"] = df["Exited"].map({0: "Retained", 1: "Churned"})

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
    df["IsActiveMember"] * 25 +
    np.where(df["NumOfProducts"] >= 2, 25, 10) +
    np.where(df["HasCrCard"] == 1, 15, 0) +
    np.where(df["Tenure"] >= 5, 20, 5) +
    np.where(df["Balance"] > 100000, 15, 5)
)

df["Relationship_Level"] = pd.cut(
    df["Relationship_Score"],
    bins=[0, 45, 75, 120],
    labels=["Weak", "Medium", "Strong"]
)


# ---------------- ML MODEL ----------------
@st.cache_resource
def train_model(data):
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
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return model, geo_encoder, gender_encoder, features, accuracy, auc


model, geo_encoder, gender_encoder, features, accuracy, auc = train_model(df)

ml_df = df.copy()
ml_df["Geography_Encoded"] = geo_encoder.transform(ml_df["Geography"])
ml_df["Gender_Encoded"] = gender_encoder.transform(ml_df["Gender"])
df["Churn_Probability"] = model.predict_proba(ml_df[features])[:, 1]

df["ML_Risk_Level"] = pd.cut(
    df["Churn_Probability"],
    bins=[0, 0.35, 0.65, 1],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

df["Churn_Probability_Percent"] = (df["Churn_Probability"] * 100).round(2)


# ---------------- SIDEBAR ----------------
st.sidebar.title("🏦 Control Center")

refresh_seconds = st.sidebar.slider(
    "Dashboard Refresh Speed",
    min_value=5,
    max_value=60,
    value=15,
    step=5
)

if AUTOREFRESH_AVAILABLE:
    refresh_count = st_autorefresh(
        interval=refresh_seconds * 1000,
        key="bank_realtime_refresh"
    )
else:
    refresh_count = 0
    st.sidebar.info("For auto refresh, add streamlit-autorefresh in requirements.txt")

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
    "Select ML Risk Level",
    ["Low Risk", "Medium Risk", "High Risk"],
    default=["Low Risk", "Medium Risk", "High Risk"]
)

active_filter = st.sidebar.multiselect(
    "Active Member Status",
    [0, 1],
    default=[0, 1],
    format_func=lambda x: "Active" if x == 1 else "Inactive"
)

filtered_df = df[
    (df["Geography"].isin(geography_filter)) &
    (df["Gender"].isin(gender_filter)) &
    (df["ML_Risk_Level"].astype(str).isin(risk_filter)) &
    (df["IsActiveMember"].isin(active_filter))
]


# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🏦 Real-Time Bank Customer Retention Intelligence Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Live-style customer churn monitoring • ML risk scoring • retention strategy • executive analytics</div>",
    unsafe_allow_html=True
)

last_updated = datetime.now().strftime("%d %b %Y, %I:%M:%S %p")
st.caption(f"Last refreshed: {last_updated}")


# ---------------- KPI CALCULATIONS ----------------
total_customers = len(filtered_df)
churned_customers = int(filtered_df["Exited"].sum())
retained_customers = total_customers - churned_customers
churn_rate = round((churned_customers / total_customers) * 100, 2) if total_customers > 0 else 0
avg_balance = round(filtered_df["Balance"].mean(), 2) if total_customers > 0 else 0
high_risk_customers = filtered_df[filtered_df["ML_Risk_Level"].astype(str) == "High Risk"]
high_risk_count = len(high_risk_customers)


# ---------------- KPI CARDS ----------------
k1, k2, k3, k4, k5, k6 = st.columns(6)

kpi_data = [
    (k1, "Total Customers", f"{total_customers:,}"),
    (k2, "Churn Rate", f"{churn_rate}%"),
    (k3, "Retained Customers", f"{retained_customers:,}"),
    (k4, "Churned Customers", f"{churned_customers:,}"),
    (k5, "High Risk Customers", f"{high_risk_count:,}"),
    (k6, "Avg Balance", f"€{avg_balance:,.0f}")
]

for col, label, value in kpi_data:
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


# ---------------- ALERT BOX ----------------
if churn_rate >= 25:
    st.markdown(
        f"""
        <div class='alert-high'>
        🚨 Critical Alert: Current churn rate is {churn_rate}%. Immediate retention campaign is required.
        </div>
        """,
        unsafe_allow_html=True
    )
elif churn_rate >= 15:
    st.markdown(
        f"""
        <div class='alert-medium'>
        ⚠️ Warning: Churn rate is {churn_rate}%. Monitor inactive and high-balance customers.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <div class='alert-good'>
        ✅ Stable: Churn rate is {churn_rate}%. Continue monitoring customer engagement.
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Live Overview",
    "🚨 Risk Command Center",
    "👥 Customer Segments",
    "🤖 ML Prediction",
    "📈 Business Insights",
    "📋 Data Explorer"
])


# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("📊 Real-Time Executive Overview")

    c1, c2 = st.columns(2)

    with c1:
        status_df = filtered_df["Customer_Status"].value_counts().reset_index()
        status_df.columns = ["Status", "Count"]

        fig = px.pie(
            status_df,
            names="Status",
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
        risk_df = filtered_df["ML_Risk_Level"].value_counts().reset_index()
        risk_df.columns = ["Risk Level", "Customers"]

        fig = px.bar(
            risk_df,
            x="Risk Level",
            y="Customers",
            text="Customers",
            title="ML Risk Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        active_churn = filtered_df.groupby("IsActiveMember")["Exited"].mean().reset_index()
        active_churn["Member Type"] = active_churn["IsActiveMember"].map({0: "Inactive", 1: "Active"})
        active_churn["Churn Rate (%)"] = (active_churn["Exited"] * 100).round(2)

        fig = px.bar(
            active_churn,
            x="Member Type",
            y="Churn Rate (%)",
            text="Churn Rate (%)",
            title="Active vs Inactive Customer Churn"
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("🚨 Risk Command Center")

    st.markdown("### Top 25 High-Risk Customers")

    high_risk_table = filtered_df.sort_values(
        "Churn_Probability",
        ascending=False
    ).head(25)

    st.dataframe(
        high_risk_table[
            [
                "CustomerId",
                "Surname",
                "Geography",
                "Gender",
                "Age",
                "Balance",
                "NumOfProducts",
                "IsActiveMember",
                "Churn_Probability_Percent",
                "ML_Risk_Level",
                "Exited"
            ]
        ],
        use_container_width=True
    )

    csv = high_risk_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download High-Risk Customer List",
        data=csv,
        file_name="high_risk_bank_customers.csv",
        mime="text/csv"
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.scatter(
            filtered_df,
            x="Age",
            y="Balance",
            color="ML_Risk_Level",
            size="Churn_Probability_Percent",
            hover_data=["CustomerId", "Surname", "Geography", "NumOfProducts"],
            title="Customer Risk Radar: Age vs Balance"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        premium_risk = filtered_df[
            (filtered_df["Balance"] > 100000) &
            (filtered_df["IsActiveMember"] == 0)
        ]

        premium_geo = premium_risk.groupby("Geography")["CustomerId"].count().reset_index()
        premium_geo.columns = ["Geography", "Premium Risk Customers"]

        fig = px.bar(
            premium_geo,
            x="Geography",
            y="Premium Risk Customers",
            text="Premium Risk Customers",
            title="Inactive High-Balance Customers by Geography"
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("👥 Customer Segment Intelligence")

    c1, c2 = st.columns(2)

    with c1:
        segment_churn = filtered_df.groupby("Engagement_Segment")["Exited"].mean().reset_index()
        segment_churn["Churn Rate (%)"] = (segment_churn["Exited"] * 100).round(2)

        fig = px.bar(
            segment_churn,
            x="Engagement_Segment",
            y="Churn Rate (%)",
            text="Churn Rate (%)",
            title="Churn Rate by Engagement Segment"
        )
        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        product_churn = filtered_df.groupby("NumOfProducts")["Exited"].mean().reset_index()
        product_churn["Churn Rate (%)"] = (product_churn["Exited"] * 100).round(2)

        fig = px.bar(
            product_churn,
            x="NumOfProducts",
            y="Churn Rate (%)",
            text="Churn Rate (%)",
            title="Product Usage vs Churn"
        )
        st.plotly_chart(fig, use_container_width=True)

    relationship_summary = filtered_df.groupby("Relationship_Level").agg(
        Customers=("CustomerId", "count"),
        Avg_Balance=("Balance", "mean"),
        Avg_Relationship_Score=("Relationship_Score", "mean"),
        Churn_Rate=("Exited", "mean")
    ).reset_index()

    relationship_summary["Avg_Balance"] = relationship_summary["Avg_Balance"].round(2)
    relationship_summary["Avg_Relationship_Score"] = relationship_summary["Avg_Relationship_Score"].round(2)
    relationship_summary["Churn_Rate"] = (relationship_summary["Churn_Rate"] * 100).round(2)

    st.markdown("### Relationship Strength Summary")
    st.dataframe(relationship_summary, use_container_width=True)


# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("🤖 Live ML Churn Prediction")

    st.markdown(
        f"""
        Model Used: **Gradient Boosting Classifier**  
        Accuracy: **{round(accuracy * 100, 2)}%**  
        AUC Score: **{round(auc, 3)}**
        """
    )

    st.markdown("### Predict New Customer Churn Risk")

    with st.form("customer_prediction_form"):
        p1, p2, p3 = st.columns(3)

        with p1:
            credit_score = st.number_input("Credit Score", 300, 900, 650)
            age = st.number_input("Age", 18, 100, 40)
            tenure = st.number_input("Tenure", 0, 10, 5)

        with p2:
            balance = st.number_input("Balance", 0.0, 300000.0, 75000.0)
            products = st.selectbox("Number of Products", [1, 2, 3, 4])
            salary = st.number_input("Estimated Salary", 0.0, 250000.0, 100000.0)

        with p3:
            geography = st.selectbox("Geography", sorted(df["Geography"].unique()))
            gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
            has_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            active = st.selectbox("Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        input_df = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography_Encoded": geo_encoder.transform([geography])[0],
            "Gender_Encoded": gender_encoder.transform([gender])[0],
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": has_card,
            "IsActiveMember": active,
            "EstimatedSalary": salary
        }])

        probability = model.predict_proba(input_df)[0][1]
        probability_percent = round(probability * 100, 2)

        if probability >= 0.65:
            st.error(f"High Churn Risk: {probability_percent}%")
            st.write("Suggested action: Assign relationship manager, offer loyalty benefit, and call customer within 24 hours.")
        elif probability >= 0.35:
            st.warning(f"Medium Churn Risk: {probability_percent}%")
            st.write("Suggested action: Send personalized product offer and monitor engagement.")
        else:
            st.success(f"Low Churn Risk: {probability_percent}%")
            st.write("Suggested action: Continue normal engagement and cross-sell suitable products.")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability_percent,
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

        st.plotly_chart(fig, use_container_width=True)


# ---------------- TAB 5 ----------------
with tab5:
    st.subheader("📈 Business Insights & Retention Strategy")

    st.markdown("""
    ### Key Findings

    **1. Inactive customers need immediate attention**  
    Customers who are not active members usually show higher churn behavior.

    **2. High-balance inactive customers are silent risk customers**  
    These customers look valuable because they have high balance, but they may leave if the bank does not engage them.

    **3. Product usage is important**  
    Single-product customers should be targeted with cross-sell offers.

    **4. ML risk score helps the bank prioritize customers**  
    Instead of calling every customer, the bank can first target high-risk customers.

    ### Recommended Retention Actions

    - Call high-risk customers personally.
    - Offer loyalty rewards to inactive high-balance customers.
    - Cross-sell products to single-product customers.
    - Create geography-specific churn campaigns.
    - Monitor churn probability every week.
    - Use this dashboard as an early warning system.
    """)


# ---------------- TAB 6 ----------------
with tab6:
    st.subheader("📋 Full Data Explorer")

    st.dataframe(
        filtered_df[
            [
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
        ],
        use_container_width=True
    )
