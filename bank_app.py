import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Bank Customer Retention Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ---------- CSS UI DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #0f172a, #1e293b);
    color: white;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
}
.subtitle {
    font-size: 18px;
    color: #cbd5e1;
}
.kpi-card {
    background: rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}
.kpi-label {
    color: #cbd5e1;
    font-size: 14px;
}
.kpi-value {
    color: #38bdf8;
    font-size: 32px;
    font-weight: 800;
}
.insight-box {
    background: rgba(56,189,248,0.12);
    padding: 18px;
    border-left: 5px solid #38bdf8;
    border-radius: 12px;
}
[data-testid="stSidebar"] {
    background-color: #020617;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv")

df = load_data()

# ---------- FEATURE ENGINEERING ----------
df["Status"] = df["Exited"].map({0: "Retained", 1: "Churned"})
df["Active_Status"] = df["IsActiveMember"].map({0: "Inactive", 1: "Active"})

df["Risk_Level"] = np.select(
    [
        (df["Exited"] == 1) & (df["IsActiveMember"] == 0),
        (df["Exited"] == 1),
        (df["Exited"] == 0)
    ],
    ["High Risk", "Medium Risk", "Low Risk"],
    default="Unknown"
)

df["Relationship_Score"] = (
    df["IsActiveMember"]
    + df["HasCrCard"]
    + (df["NumOfProducts"] >= 2).astype(int)
    + (df["Tenure"] >= 5).astype(int)
)

df["Relationship_Level"] = pd.cut(
    df["Relationship_Score"],
    bins=[-1, 1, 2, 4],
    labels=["Weak", "Medium", "Strong"]
)

# ---------- SIDEBAR ----------
st.sidebar.title("🏦 Bank Dashboard")
st.sidebar.caption("Customer Retention Intelligence")

geo = st.sidebar.multiselect(
    "Select Geography",
    df["Geography"].unique(),
    default=df["Geography"].unique()
)

gender = st.sidebar.multiselect(
    "Select Gender",
    df["Gender"].unique(),
    default=df["Gender"].unique()
)

risk = st.sidebar.multiselect(
    "Select Risk Level",
    df["Risk_Level"].unique(),
    default=df["Risk_Level"].unique()
)

filtered_df = df[
    (df["Geography"].isin(geo)) &
    (df["Gender"].isin(gender)) &
    (df["Risk_Level"].isin(risk))
]

# ---------- HEADER ----------
st.markdown('<div class="main-title">🏦 Customer Retention Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Banking Analytics · Churn Risk · Product Utilization · Customer Relationship Strength</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------- KPI CARDS ----------
total_customers = len(filtered_df)
churned = int(filtered_df["Exited"].sum())
retained = total_customers - churned
churn_rate = round((churned / total_customers) * 100, 2) if total_customers > 0 else 0
avg_balance = round(filtered_df["Balance"].mean(), 2)

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
        <div class="kpi-label">Retained</div>
        <div class="kpi-value">{retained:,}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Churned</div>
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
        <div class="kpi-label">Avg Balance</div>
        <div class="kpi-value">€{avg_balance:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------- TABS ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🧠 Customer Behavior",
    "⚠️ Risk Analysis",
    "🎯 Retention Strategy"
])

# ---------- OVERVIEW ----------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        churn_count = filtered_df["Status"].value_counts().reset_index()
        churn_count.columns = ["Status", "Count"]

        fig = px.pie(
            churn_count,
            names="Status",
            values="Count",
            hole=0.55,
            title="Customer Churn Distribution",
            color="Status",
            color_discrete_map={
                "Retained": "#22c55e",
                "Churned": "#ef4444"
            }
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
            title="Churn Rate by Geography",
            color="Churn Rate",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"]
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>Insight:</b> This dashboard helps banks understand churn patterns using customer behavior,
    product usage, active membership, and relationship strength.
    </div>
    """, unsafe_allow_html=True)

# ---------- CUSTOMER BEHAVIOR ----------
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        active_churn = filtered_df.groupby("Active_Status")["Exited"].mean().reset_index()
        active_churn["Churn Rate"] = active_churn["Exited"] * 100

        fig = px.bar(
            active_churn,
            x="Active_Status",
            y="Churn Rate",
            text=active_churn["Churn Rate"].round(1).astype(str) + "%",
            title="Active vs Inactive Churn Rate",
            color="Active_Status",
            color_discrete_map={
                "Active": "#22c55e",
                "Inactive": "#ef4444"
            }
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
            title="Number of Products vs Churn",
            color="Churn Rate",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"]
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ---------- RISK ANALYSIS ----------
with tab3:
    risk_count = filtered_df["Risk_Level"].value_counts().reset_index()
    risk_count.columns = ["Risk Level", "Customers"]

    fig = px.bar(
        risk_count,
        x="Risk Level",
        y="Customers",
        text="Customers",
        title="Customer Risk Segmentation",
        color="Risk Level",
        color_discrete_map={
            "Low Risk": "#22c55e",
            "Medium Risk": "#f59e0b",
            "High Risk": "#ef4444"
        }
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("High Risk Customer Table")
    high_risk_df = filtered_df[filtered_df["Risk_Level"] == "High Risk"]

    st.dataframe(
        high_risk_df[
            [
                "CustomerId", "Surname", "Geography", "Gender", "Age",
                "Balance", "NumOfProducts", "IsActiveMember",
                "Exited", "Risk_Level"
            ]
        ],
        use_container_width=True
    )

# ---------- STRATEGY ----------
with tab4:
    st.markdown("""
    <div class="insight-box">
    <h3>🎯 Client Retention Recommendations</h3>
    <b>1. Focus on inactive customers:</b> Inactive customers show higher churn risk.<br><br>
    <b>2. Improve product engagement:</b> Customers with low product usage should receive cross-sell offers.<br><br>
    <b>3. Protect high-balance customers:</b> High-value customers need personalized relationship manager support.<br><br>
    <b>4. Monitor geography-wise churn:</b> Region-based retention campaigns can improve performance.<br><br>
    <b>5. Build monthly churn scoring:</b> Use customer activity and product behavior to identify risk early.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Customer Data Preview")
    st.dataframe(filtered_df, use_container_width=True)
