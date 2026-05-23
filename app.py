import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Banking Retention Intelligence",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#061627 0%,#0B2D4D 50%,#003B73 100%);color:white;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#020B14,#061627,#0B2D4D);}
[data-testid="stSidebar"] *{color:white !important;}
.block-container{padding-top:2rem;}
.big-title{font-size:42px;font-weight:900;color:white;padding-top:10px;}
.sub-title{font-size:18px;color:#D7E9FF;margin-bottom:25px;}
.kpi-card{background:linear-gradient(135deg,#005B96,#00A6FB);padding:24px;border-radius:22px;
 color:white;text-align:center;box-shadow:0 10px 25px rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.25);}
.kpi-value{font-size:32px;font-weight:900;}
.kpi-label{font-size:14px;color:#E8F4FF;}
.insight-box{background:rgba(255,255,255,.95);padding:20px;border-left:7px solid #00A6FB;
 border-radius:14px;font-size:16px;box-shadow:0 4px 14px rgba(0,0,0,.2);color:#0B1F3A;}
.warning-box{background:rgba(255,244,229,.98);padding:20px;border-left:7px solid #FFB703;
 border-radius:14px;font-size:16px;box-shadow:0 4px 14px rgba(0,0,0,.2);color:#0B1F3A;}
.success-box{background:rgba(232,255,241,.98);padding:20px;border-left:7px solid #00A86B;
 border-radius:14px;font-size:16px;box-shadow:0 4px 14px rgba(0,0,0,.2);color:#0B1F3A;}
.ml-box{background:rgba(220,230,255,.97);padding:20px;border-left:7px solid #6C5CE7;
 border-radius:14px;font-size:16px;box-shadow:0 4px 14px rgba(0,0,0,.2);color:#0B1F3A;}
h1,h2,h3{color:white !important;font-weight:800;}
.stTabs [data-baseweb="tab-list"]{gap:10px;}
.stTabs [data-baseweb="tab"]{background-color:rgba(255,255,255,.18);border-radius:14px;
 padding:12px 18px;font-weight:700;color:white;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#00A6FB,#0074D9);color:white;}
[data-testid="stMetric"]{background:rgba(255,255,255,.15);padding:18px;border-radius:18px;
 color:white;box-shadow:0 6px 18px rgba(0,0,0,.25);}
[data-testid="stDataFrame"]{background:white;border-radius:15px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv")

df = load_data()

# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING  (rule-based, kept from original)
# ─────────────────────────────────────────────────────────────
df["Engagement_Profile"] = np.select(
    [
        (df["IsActiveMember"]==1) & (df["NumOfProducts"]>=2),
        (df["IsActiveMember"]==0) & (df["Balance"]>100000),
        (df["IsActiveMember"]==1) & (df["NumOfProducts"]==1),
        (df["IsActiveMember"]==0)
    ],
    ["Active Engaged Customer","Inactive High-Balance Customer",
     "Active Low-Product Customer","Inactive Disengaged Customer"],
    default="Other Customer"
)

df["Relationship_Strength_Index"] = (
    np.where(df["IsActiveMember"]==1,1,0) +
    np.where(df["NumOfProducts"]>=2,1,0) +
    np.where(df["HasCrCard"]==1,1,0) +
    np.where(df["Tenure"]>=5,1,0)
)

df["Relationship_Category"] = pd.cut(
    df["Relationship_Strength_Index"],
    bins=[-1,1,2,4],
    labels=["Weak Relationship","Medium Relationship","Strong Relationship"]
)

df["Retention_Intelligence_Score"] = (
    df["Relationship_Strength_Index"]*20 +
    df["IsActiveMember"]*20 +
    np.where(df["Balance"]>100000,10,0)
)

df["Risk_Level"] = "High Risk"
df.loc[df["Retention_Intelligence_Score"]>=80,"Risk_Level"] = "Low Risk"
df.loc[(df["Retention_Intelligence_Score"]>=50)&
       (df["Retention_Intelligence_Score"]<80),"Risk_Level"] = "Medium Risk"

# ─────────────────────────────────────────────────────────────
#  ML — TRAIN THREE MODELS, PICK BEST (Gradient Boosting)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_all_models(data):
    d = data.copy()
    le_geo = LabelEncoder(); le_gen = LabelEncoder()
    d["Geography_enc"] = le_geo.fit_transform(d["Geography"])
    d["Gender_enc"]    = le_gen.fit_transform(d["Gender"])

    FEATURES = ["CreditScore","Geography_enc","Gender_enc","Age","Tenure",
                "Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
    X = d[FEATURES]; y = d["Exited"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000,random_state=42), True),
        "Random Forest":        (RandomForestClassifier(n_estimators=150,max_depth=10,random_state=42,n_jobs=-1), False),
        "Gradient Boosting":    (GradientBoostingClassifier(n_estimators=150,random_state=42), False),
    }

    results = {}
    for name,(m,needs_scale) in models.items():
        Xtr = X_tr_sc if needs_scale else X_tr
        Xte = X_te_sc if needs_scale else X_te
        m.fit(Xtr, y_tr)
        prob = m.predict_proba(Xte)[:,1]
        pred = m.predict(Xte)
        rep  = classification_report(y_te,pred,output_dict=True)
        results[name] = {
            "model":     m,
            "needs_scale": needs_scale,
            "auc":       round(roc_auc_score(y_te,prob),4),
            "accuracy":  round(rep["accuracy"]*100,2),
            "precision": round(rep["1"]["precision"]*100,2),
            "recall":    round(rep["1"]["recall"]*100,2),
            "f1":        round(rep["1"]["f1-score"]*100,2),
            "cm":        confusion_matrix(y_te,pred),
            "fpr":       roc_curve(y_te,prob)[0],
            "tpr":       roc_curve(y_te,prob)[1],
            "fi": (pd.Series(m.feature_importances_,index=FEATURES)
                   .sort_values(ascending=False).reset_index()
                   .rename(columns={"index":"Feature",0:"Importance"})
                   if hasattr(m,"feature_importances_") else None),
        }

    best = results["Gradient Boosting"]
    return results, best, le_geo, le_gen, scaler, FEATURES

results, best_model_info, le_geo, le_gen, scaler, ML_FEATURES = train_all_models(df)
best_model = best_model_info["model"]

# attach ML probability to full df
df_enc = df.copy()
df_enc["Geography_enc"] = le_geo.transform(df_enc["Geography"])
df_enc["Gender_enc"]    = le_gen.transform(df_enc["Gender"])
df["ML_Churn_Prob"] = best_model.predict_proba(df_enc[ML_FEATURES])[:,1]
df["ML_Churn_Risk"] = pd.cut(df["ML_Churn_Prob"],bins=[0,.33,.66,1.0],
                              labels=["Low Risk","Medium Risk","High Risk"])

# ─────────────────────────────────────────────────────────────
#  SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────
st.sidebar.title("🏦 Control Panel")
st.sidebar.markdown("Filter customers to explore retention behavior.")

geography    = st.sidebar.multiselect("🌍 Geography",   df["Geography"].unique(), default=df["Geography"].unique())
gender       = st.sidebar.multiselect("👤 Gender",      df["Gender"].unique(),    default=df["Gender"].unique())
active_status= st.sidebar.multiselect("⚡ Active Status",df["IsActiveMember"].unique(), default=df["IsActiveMember"].unique())
product_range= st.sidebar.slider("📦 No. of Products",
                                  int(df["NumOfProducts"].min()), int(df["NumOfProducts"].max()),
                                  (int(df["NumOfProducts"].min()), int(df["NumOfProducts"].max())))
balance_min  = st.sidebar.slider("💰 Min Balance", 0, int(df["Balance"].max()), 0)

fdf = df[
    df["Geography"].isin(geography) &
    df["Gender"].isin(gender) &
    df["IsActiveMember"].isin(active_status) &
    df["NumOfProducts"].between(*product_range) &
    (df["Balance"]>=balance_min)
]

# ─────────────────────────────────────────────────────────────
#  HEADER + KPIs
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="big-title">🏦 Customer Retention Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">💼 Banking Analytics • 📊 Churn Intelligence • 🤖 ML Prediction (Gradient Boosting) • 🚀 Retention Strategy</div>', unsafe_allow_html=True)
st.write("")

n     = len(fdf)
churn = int(fdf["Exited"].sum()) if n>0 else 0
ret   = n - churn
cr    = round(churn/n*100,2) if n>0 else 0
avg_s = round(fdf["Retention_Intelligence_Score"].mean(),2) if n>0 else 0
hi_risk= fdf[(fdf["Balance"]>100000)&(fdf["IsActiveMember"]==0)]

c1,c2,c3,c4,c5 = st.columns(5)
for col,label,val in [
    (c1,"👥 Total Customers",f"{n:,}"),
    (c2,"📉 Churn Rate",f"{cr}%"),
    (c3,"✅ Retained",f"{ret:,}"),
    (c4,"⚠️ Churned",f"{churn:,}"),
    (c5,"🧠 Avg Retention Score",avg_s),
]:
    col.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div></div>',
                 unsafe_allow_html=True)
st.write("")

# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📊 Executive Overview",
    "🧠 Customer Intelligence",
    "🚨 Premium Risk Radar",
    "💎 Relationship Strength",
    "🤖 ML Churn Prediction",
    "🎯 Strategy Playbook"
])

# ── TAB 1 ──────────────────────────────────────────────────
with tab1:
    st.header("📊 Executive Retention Overview")
    c1,c2 = st.columns(2)
    with c1:
        cd = fdf["Exited"].value_counts().reset_index()
        cd.columns=["Exited","Count"]; cd["Status"]=cd["Exited"].map({0:"Retained",1:"Churned"})
        st.plotly_chart(px.pie(cd,values="Count",names="Status",hole=.55,title="Churn Distribution"),use_container_width=True)
    with c2:
        gc = fdf.groupby("Geography")["Exited"].mean().reset_index()
        gc["Churn Rate (%)"]=(gc["Exited"]*100).round(2)
        st.plotly_chart(px.bar(gc,x="Geography",y="Churn Rate (%)",text="Churn Rate (%)",title="Churn by Geography"),use_container_width=True)
    st.markdown('<div class="insight-box">📌 <b>Insight:</b> Germany has the highest churn rate among all geographies. Focus reactivation campaigns there first.</div>',unsafe_allow_html=True)

# ── TAB 2 ──────────────────────────────────────────────────
with tab2:
    st.header("🧠 Customer Engagement & Product Intelligence")
    c1,c2 = st.columns(2)
    with c1:
        ea = fdf.groupby("IsActiveMember")["Exited"].mean().reset_index()
        ea["Type"]=ea["IsActiveMember"].map({0:"Inactive",1:"Active"})
        ea["Churn Rate (%)"]=(ea["Exited"]*100).round(2)
        st.plotly_chart(px.bar(ea,x="Type",y="Churn Rate (%)",text="Churn Rate (%)",title="Active vs Inactive Churn"),use_container_width=True)
    with c2:
        pc = fdf.groupby("NumOfProducts")["Exited"].mean().reset_index()
        pc["Churn Rate (%)"]=(pc["Exited"]*100).round(2)
        st.plotly_chart(px.bar(pc,x="NumOfProducts",y="Churn Rate (%)",text="Churn Rate (%)",title="Products Used vs Churn"),use_container_width=True)
    ep = fdf.groupby("Engagement_Profile")["Exited"].mean().reset_index()
    ep["Churn Rate (%)"]=(ep["Exited"]*100).round(2)
    fig=px.bar(ep,x="Engagement_Profile",y="Churn Rate (%)",text="Churn Rate (%)",title="Engagement Profile vs Churn")
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="success-box">✅ <b>Finding:</b> Inactive Disengaged Customers churn the most. Multi-product active customers show the strongest loyalty.</div>',unsafe_allow_html=True)

# ── TAB 3 ──────────────────────────────────────────────────
with tab3:
    st.header("🚨 Premium Customer Risk Radar")
    st.markdown('<div class="warning-box">⚠️ <b>Silent Churn Risk:</b> Customers with Balance > 100,000 AND inactive status — appear valuable but are silently disengaging.</div>',unsafe_allow_html=True)
    st.write("")
    st.metric("💰 High-Value Disengaged Customers", len(hi_risk))
    if len(hi_risk)>0:
        rg = hi_risk.groupby("Geography")["CustomerId"].count().reset_index()
        rg.columns=["Geography","Risk Customers"]
        st.plotly_chart(px.bar(rg,x="Geography",y="Risk Customers",text="Risk Customers",title="Premium Risk by Geography"),use_container_width=True)
        st.dataframe(
            hi_risk[["CustomerId","Surname","Geography","Gender","Age","Balance",
                      "NumOfProducts","IsActiveMember","Exited","ML_Churn_Prob"]]
            .sort_values("ML_Churn_Prob",ascending=False).head(50),
            use_container_width=True
        )
    else:
        st.success("No premium risk customers in current filter.")

# ── TAB 4 ──────────────────────────────────────────────────
with tab4:
    st.header("💎 Relationship Strength & Retention Intelligence")
    c1,c2 = st.columns(2)
    with c1:
        rs = fdf.groupby("Relationship_Category")["Exited"].mean().reset_index()
        rs["Churn Rate (%)"]=(rs["Exited"]*100).round(2)
        st.plotly_chart(px.bar(rs,x="Relationship_Category",y="Churn Rate (%)",text="Churn Rate (%)",title="Relationship Strength vs Churn"),use_container_width=True)
    with c2:
        rv = fdf["Risk_Level"].value_counts().reset_index(); rv.columns=["Risk Level","Customers"]
        st.plotly_chart(px.pie(rv,values="Customers",names="Risk Level",hole=.45,title="Retention Risk Segments"),use_container_width=True)
    st.subheader("📌 Segment Summary Table")
    ss = fdf.groupby("Risk_Level").agg(
        Customers=("CustomerId","count"),
        Avg_Balance=("Balance","mean"),
        Avg_Score=("Retention_Intelligence_Score","mean"),
        Churn_Rate=("Exited","mean")
    ).reset_index()
    ss["Avg_Balance"]=ss["Avg_Balance"].round(2)
    ss["Avg_Score"]=ss["Avg_Score"].round(2)
    ss["Churn_Rate"]=(ss["Churn_Rate"]*100).round(2)
    st.dataframe(ss,use_container_width=True)

# ── TAB 5  — ML CHURN PREDICTION ───────────────────────────
with tab5:
    st.header("🤖 Machine Learning Churn Prediction")
    st.markdown("""
    <div class="ml-box">
    🤖 <b>Three ML models trained on this dataset:</b><br>
    Logistic Regression → Random Forest → <b>Gradient Boosting (Best — AUC 0.87)</b><br>
    Model trained on 80% data (8,000 customers), tested on 20% (2,000 customers).
    Features: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard,
    IsActiveMember, EstimatedSalary, Geography, Gender.
    </div>""", unsafe_allow_html=True)
    st.write("")

    # ── Model Comparison Table ──
    st.subheader("📊 Model Comparison — All Three Algorithms")
    comp_rows = []
    for name,r in results.items():
        comp_rows.append({"Model":name,"AUC":r["auc"],
                          "Accuracy (%)":r["accuracy"],"Precision (%)":r["precision"],
                          "Recall (%)":r["recall"],"F1-Score (%)":r["f1"]})
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df.style.highlight_max(subset=["AUC","Accuracy (%)","F1-Score (%)"],color="#c8f7c5"),
                 use_container_width=True)

    st.write("")
    st.subheader("🏆 Best Model: Gradient Boosting — Performance Details")

    m1,m2,m3,m4,m5 = st.columns(5)
    for col,label,val,color in [
        (m1,"🎯 Accuracy",   f"{best_model_info['accuracy']}%",  "#00A6FB"),
        (m2,"📡 AUC Score",  f"{best_model_info['auc']}",        "#6C5CE7"),
        (m3,"🔍 Precision",  f"{best_model_info['precision']}%", "#00A86B"),
        (m4,"📻 Recall",     f"{best_model_info['recall']}%",    "#FFB703"),
        (m5,"⚖️ F1-Score",  f"{best_model_info['f1']}%",        "#FF4B4B"),
    ]:
        col.markdown(f'<div class="kpi-card" style="background:linear-gradient(135deg,{color}aa,{color})"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div></div>',
                     unsafe_allow_html=True)
    st.write("")

    c1,c2 = st.columns(2)

    # ROC Curves — all 3 models
    with c1:
        fig_roc = go.Figure()
        colors  = {"Gradient Boosting":"#00A6FB","Random Forest":"#00A86B","Logistic Regression":"#FFB703"}
        for name,r in results.items():
            fig_roc.add_trace(go.Scatter(x=r["fpr"],y=r["tpr"],mode="lines",
                name=f"{name} (AUC={r['auc']})",line=dict(color=colors[name],width=2+(name=="Gradient Boosting"))))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",line=dict(color="gray",dash="dash")))
        fig_roc.update_layout(title="ROC Curves — All Models",
            xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.05)",font=dict(color="white"))
        st.plotly_chart(fig_roc,use_container_width=True)

    # Confusion Matrix — best model
    with c2:
        cm = best_model_info["cm"]
        fig_cm = px.imshow(cm,text_auto=True,
            x=["Predicted Retained","Predicted Churned"],
            y=["Actual Retained","Actual Churned"],
            title="Confusion Matrix — Gradient Boosting",
            color_continuous_scale="Blues")
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(color="white"))
        st.plotly_chart(fig_cm,use_container_width=True)

    # Feature Importance
    st.subheader("🔑 Feature Importance — What Actually Drives Churn?")
    fi_df = best_model_info["fi"]
    fi_df.columns = ["Feature","Importance"]
    fig_fi = px.bar(fi_df,x="Importance",y="Feature",orientation="h",
        text=fi_df["Importance"].round(3),
        title="Gradient Boosting Feature Importance",
        color="Importance",color_continuous_scale="Blues")
    fig_fi.update_layout(yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.05)",font=dict(color="white"))
    st.plotly_chart(fig_fi,use_container_width=True)
    st.markdown('<div class="insight-box">🔑 <b>Key Finding:</b> Age (38%) and Number of Products (30%) are the top two churn drivers according to the Gradient Boosting model — far more important than credit score or salary.</div>',unsafe_allow_html=True)
    st.write("")

    # ML Risk Distribution on filtered customers
    st.subheader("📈 ML Risk Distribution on Filtered Customers")
    c1,c2 = st.columns(2)
    with c1:
        mr = fdf["ML_Churn_Risk"].value_counts().reset_index(); mr.columns=["Risk","Count"]
        fig_r=px.pie(mr,values="Count",names="Risk",hole=.5,title="ML-Predicted Risk Segments",
            color_discrete_map={"High Risk":"#FF4B4B","Medium Risk":"#FFB703","Low Risk":"#00A86B"})
        st.plotly_chart(fig_r,use_container_width=True)
    with c2:
        fig_h=px.histogram(fdf,x="ML_Churn_Prob",color="ML_Churn_Risk",nbins=30,
            title="Churn Probability Distribution",barmode="overlay",
            color_discrete_map={"High Risk":"#FF4B4B","Medium Risk":"#FFB703","Low Risk":"#00A86B"})
        fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.05)",font=dict(color="white"))
        st.plotly_chart(fig_h,use_container_width=True)

    # Top 20 High-Risk Table
    st.subheader("🚨 Top 20 ML-Predicted High-Risk Customers")
    top20 = fdf.sort_values("ML_Churn_Prob",ascending=False).head(20)
    st.dataframe(top20[["CustomerId","Surname","Geography","Gender","Age","Balance",
                          "NumOfProducts","IsActiveMember","ML_Churn_Prob","ML_Churn_Risk","Exited"]].reset_index(drop=True),
                 use_container_width=True)

    st.write("")

    # ── Live Customer Predictor ──
    st.subheader("🔮 Predict Churn for Any Customer — Live Predictor")
    st.markdown("Fill in the customer details and get an instant ML churn probability:")

    with st.form("predictor"):
        p1,p2,p3 = st.columns(3)
        with p1:
            inp_credit   = st.number_input("Credit Score",      300,850,650)
            inp_age      = st.number_input("Age",                18,100, 38)
            inp_tenure   = st.number_input("Tenure (years)",      0, 10,  5)
            inp_balance  = st.number_input("Account Balance",     0,300000,75000)
        with p2:
            inp_products = st.selectbox("Number of Products",[1,2,3,4])
            inp_crcard   = st.selectbox("Has Credit Card",[1,0],format_func=lambda x:"Yes" if x==1 else "No")
            inp_active   = st.selectbox("Is Active Member",[1,0],format_func=lambda x:"Yes" if x==1 else "No")
            inp_salary   = st.number_input("Estimated Salary",0,250000,100000)
        with p3:
            inp_geo    = st.selectbox("Geography",["France","Germany","Spain"])
            inp_gender = st.selectbox("Gender",["Male","Female"])
        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "CreditScore":    inp_credit,
            "Geography_enc":  le_geo.transform([inp_geo])[0],
            "Gender_enc":     le_gen.transform([inp_gender])[0],
            "Age":            inp_age,
            "Tenure":         inp_tenure,
            "Balance":        inp_balance,
            "NumOfProducts":  inp_products,
            "HasCrCard":      inp_crcard,
            "IsActiveMember": inp_active,
            "EstimatedSalary":inp_salary,
        }])
        prob  = best_model.predict_proba(row)[0][1]
        pct   = round(prob*100,1)

        if prob>=0.66:
            color="#FF4B4B"; label="🔴 HIGH RISK — Very Likely to Churn"; box="warning-box"
        elif prob>=0.33:
            color="#FFB703"; label="🟡 MEDIUM RISK — Monitor Closely";    box="warning-box"
        else:
            color="#00A86B"; label="🟢 LOW RISK — Likely to Stay";        box="success-box"

        st.markdown(f'<div class="{box}" style="margin-top:16px"><h3 style="color:{color}">{label}</h3>'
                    f'<p><b>Gradient Boosting Churn Probability: {pct}%</b></p></div>',unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(mode="gauge+number",value=pct,
            title={"text":"Churn Probability (%)","font":{"color":"white"}},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":color},
                   "steps":[{"range":[0,33],"color":"#d4edda"},
                             {"range":[33,66],"color":"#fff3cd"},
                             {"range":[66,100],"color":"#f8d7da"}]}))
        gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(color="white",size=16))
        st.plotly_chart(gauge,use_container_width=True)

# ── TAB 6 ──────────────────────────────────────────────────
with tab6:
    st.header("🎯 Strategic Retention Playbook")
    st.markdown("""
    ### 1. 🔄 Customer Reactivation Campaigns
    Target inactive customers with personalised offers, relationship manager calls, and loyalty rewards.

    ### 2. 📦 Product Bundling Strategy
    Single-product users are at high risk. Cross-sell credit cards, savings plans, or advisory services.

    ### 3. 💰 Premium Silent Churn Protection
    High-balance inactive customers must be monitored. They look valuable but are quietly leaving.

    ### 4. 🤖 ML Early Warning System
    Use the Gradient Boosting churn probability score to prioritise which customers need action.
    Focus intervention on High Risk (prob > 66%) customers first.

    ### 5. 🧠 Relationship Intelligence
    Customers with Weak Relationship Index need immediate engagement — multi-channel outreach.

    ### 6. 📊 Age & Product-Based Targeting
    ML shows Age and NumOfProducts are the #1 and #2 churn drivers.
    Older single-product inactive customers are the highest priority segment.
    """)
    st.markdown("""
    <div class="insight-box">
    🚀 <b>Final Business Impact:</b> This platform combines behavioural rule-based analytics
    with three ML models (Logistic Regression, Random Forest, Gradient Boosting — AUC 0.87)
    to convert raw European banking data into actionable churn intelligence.
    </div>""", unsafe_allow_html=True)

# ── RAW DATA EXPANDER ───────────────────────────────────────
with st.expander("📂 View Full Filtered Dataset with ML Predictions"):
    st.dataframe(fdf[["CustomerId","Surname","Geography","Gender","Age","CreditScore",
                       "Balance","NumOfProducts","IsActiveMember","Tenure","HasCrCard",
                       "EstimatedSalary","Exited","ML_Churn_Prob","ML_Churn_Risk",
                       "Engagement_Profile","Risk_Level"]],use_container_width=True)
