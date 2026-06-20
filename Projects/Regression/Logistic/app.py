import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



st.set_page_config(
    page_title="Medicare Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1E2130, #252A3D);
        border: 1px solid #2E3450;
        border-radius: 12px;
        padding: 28px 16px;
        text-align: center;
        margin: 8px 0;
        min-height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #4C9BE8;
        margin: 0;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8892A4;
        margin: 8px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    .metric-card.fraud .metric-value { color: #E84C4C; }
    .metric-card.good  .metric-value { color: #4CE89B; }
    .insight-box {
        background: #1A1F30;
        border: 1px solid #2E3450;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .insight-box.warning { border-left: 4px solid #E84C4C; }
    .insight-box.success { border-left: 4px solid #4CE89B; }
    .insight-box.info    { border-left: 4px solid #4C9BE8; }
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


CORE_FEATURES = [
    "TotalClaims", "TotalReimbursed", "AvgReimbursed",
    "AvgClaimDuration", "AvgDaysInHospital",
    "UniquePatients", "UniqueAttendPhys",
    "SameAttendOperRate", "AvgChronicConds", "InpatientRatio",
]

@st.cache_resource
def load_all():
    """Load model, scaler, data and precompute predictions once."""
    # Load saved artifacts
    model  = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Load data
    df = pd.read_csv("healthcare_fraud_features.csv")

    # Rebuild the exact same split used during training
    X = df[CORE_FEATURES].copy()
    y = df["FraudLabel"].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale using the LOADED scaler (same as training)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Predictions
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Force integer types — prevents the metrics error
    y_test  = y_test.astype(int).values
    y_pred  = y_pred.astype(int)
    y_proba = y_proba.astype(float)

    return model, scaler, df, X_test, y_test, y_pred, y_proba

# Load
try:
    model, scaler, final_df, X_test_raw, y_test, y_pred, y_proba = load_all()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Error loading files: {e}")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate to", [
    "📋  Project Overview",
    "📊  EDA Dashboard",
    "🤖  Model Performance",
    "📈  ROC & PR Curves",
    "🎚️  Threshold Tuner",
    "🔮  Live Prediction",
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Info**
- Algorithm: Logistic Regression
- Imbalance: SMOTE
- Features: 10 core
- AUC-ROC: 0.962
""")


# ════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "📋  Project Overview":

    st.title("🔍 Medicare Fraud Detection System")
    st.markdown("#### Identifying fraudulent healthcare providers using machine learning")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <p class="metric-value">5,410</p>
            <p class="metric-label">Total Providers</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card fraud">
            <p class="metric-value">506</p>
            <p class="metric-label">Fraud Providers (9.4%)</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <p class="metric-value">558K</p>
            <p class="metric-label">Total Claims</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card good">
            <p class="metric-value">0.962</p>
            <p class="metric-label">Model AUC-ROC</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🏥 What is Medicare Fraud?")
        st.markdown("""
        **Medicare** is the US government health insurance program for people aged 65+.
        Healthcare providers (hospitals and doctors) submit claims to Medicare requesting
        reimbursement for treatments given to patients.

        **Fraud occurs when providers:**
        - Bill for services never provided (phantom billing)
        - Exaggerate the severity of conditions (upcoding)
        - Submit duplicate claims for the same service
        - Bill for deceased patients
        - Keep patients admitted longer than medically necessary

        **The impact:** Medicare loses billions of dollars annually to fraudulent claims,
        directly increasing healthcare costs for everyone.
        """)

        st.markdown("### 🎯 Project Goal")
        st.markdown("""
        Predict which healthcare providers are **potentially fraudulent** based on
        patterns in the claims they submit — enabling Medicare investigators to
        prioritise their limited resources on the highest-risk providers.
        """)

    with col_right:
        st.markdown("### 📁 Dataset")
        st.markdown("""
        | File | Description | Rows |
        |---|---|---|
        | Labels | Provider + fraud label | 5,410 |
        | Beneficiary | Patient demographics | 138,556 |
        | Inpatient | Hospital stay claims | 40,474 |
        | Outpatient | Day visit claims | 517,737 |
        """)
        st.markdown("### ⚙️ Approach")
        for i, step in enumerate([
            "EDA on all 4 datasets individually",
            "Data cleaning and feature engineering",
            "Aggregated claims to provider level",
            "Stratified 80/20 train-test split",
            "StandardScaler (fit on train only)",
            "SMOTE for 9:1 class imbalance",
            "Logistic Regression baseline model",
            "Threshold tuning to optimise Recall",
        ], 1):
            st.markdown(f"**{i}.** {step}")

    st.markdown("---")
    st.markdown("### 📊 Class Distribution")
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0F1117")
        for ax in axes:
            ax.set_facecolor("#1E2130")

        bars = axes[0].bar(["Not Fraud", "Fraud"], [4904, 506],
                           color=["#4C9BE8","#E84C4C"], edgecolor="none", width=0.5)
        axes[0].set_title("Provider Count", color="white", fontsize=12)
        axes[0].tick_params(colors="white")
        axes[0].spines[:].set_visible(False)
        for bar, val, pct in zip(bars, [4904, 506], ["90.6%","9.4%"]):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                         f"{val:,}\n({pct})", ha="center", color="white",
                         fontweight="bold", fontsize=10)

        axes[1].pie([4904, 506], labels=["Not Fraud","Fraud"],
                    colors=["#4C9BE8","#E84C4C"], autopct="%1.1f%%",
                    startangle=90, textprops={"color":"white"},
                    wedgeprops={"edgecolor":"#0F1117","linewidth":2})
        axes[1].set_title("Class Split", color="white", fontsize=12)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("""<div class="insight-box warning">
    ⚠️ <b>Class Imbalance:</b> Only 9.4% of providers are fraudulent — a roughly 10:1 ratio.
    Accuracy is misleading here. We use <b>Recall</b> and <b>AUC-ROC</b> instead.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — EDA DASHBOARD
# ════════════════════════════════════════════════════════════
elif page == "📊  EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    if not MODEL_LOADED:
        st.error("Data file not found.")
        st.stop()

    st.markdown("### 🔎 Feature Distribution by Fraud Label")
    selected = st.selectbox("Select a feature:", CORE_FEATURES, index=1)

    col1, col2 = st.columns([2, 1])

    fraud_data     = final_df.loc[final_df["FraudLabel"]==1, selected].dropna()
    non_fraud_data = final_df.loc[final_df["FraudLabel"]==0, selected].dropna()

    with col1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0F1117")
        for ax in axes:
            ax.set_facecolor("#1E2130")

        # Histogram
        axes[0].hist(non_fraud_data, bins=40, alpha=0.7, density=True,
                    color="#4C9BE8", label="Not Fraud", edgecolor="none")
        axes[0].hist(fraud_data, bins=40, alpha=0.7, density=True,
                    color="#E84C4C", label="Fraud", edgecolor="none")
        axes[0].set_title(f"{selected} — Distribution",
                        color="white", fontsize=13, fontweight="bold", pad=12)
        axes[0].set_xlabel(selected, color="white", fontsize=11)
        axes[0].set_ylabel("Density", color="white", fontsize=11)
        axes[0].legend(facecolor="#1E2130", labelcolor="white", fontsize=11)
        axes[0].tick_params(colors="white", labelsize=10)
        axes[0].spines[:].set_visible(False)

        # Box plot
        bp = axes[1].boxplot(
            [non_fraud_data, fraud_data],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(color="#8892A4", linewidth=1.5),
            capprops=dict(color="#8892A4", linewidth=1.5),
            flierprops=dict(marker="o", color="#8892A4",
                            markersize=4, alpha=0.5)
        )
        bp["boxes"][0].set_facecolor("#4C9BE8")
        bp["boxes"][0].set_alpha(0.8)
        bp["boxes"][1].set_facecolor("#E84C4C")
        bp["boxes"][1].set_alpha(0.8)

        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(["Not Fraud", "Fraud"],
                                color="white", fontsize=12)
        axes[1].set_title(f"{selected} — Box Plot",
                        color="white", fontsize=13,
                        fontweight="bold", pad=12)
        axes[1].tick_params(colors="white", labelsize=10)
        axes[1].spines[:].set_visible(False)

        # Median labels — bigger and more visible
        for j, grp in enumerate([non_fraud_data, fraud_data]):
            axes[1].text(
                j + 1, grp.median(),
                f"  {grp.median():,.1f}",
                va="center", color="white",
                fontsize=11, fontweight="bold"
            )

        plt.tight_layout(pad=2.0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### Stats Comparison")
        stats = pd.DataFrame({
            "Not Fraud": non_fraud_data.describe(),
            "Fraud"    : fraud_data.describe()
        }).round(2)
        st.dataframe(stats, use_container_width=True)
        denom = non_fraud_data.median() if non_fraud_data.median() != 0 else 1
        ratio = fraud_data.median() / denom
        if ratio > 1.5:
            st.markdown(f"""<div class="insight-box warning">
            🚨 Fraud providers have <b>{ratio:.1f}x higher</b> median {selected}.
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Fraud Provider Profile")
    profile = (
        final_df.groupby("FraudLabel")[CORE_FEATURES].median().T
        .rename(columns={0:"Not Fraud (median)", 1:"Fraud (median)"})
    )
    profile["Ratio"] = (
        profile["Fraud (median)"] /
        profile["Not Fraud (median)"].replace(0, np.nan)
    ).round(2)
    st.dataframe(profile.round(3), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔥 Correlation Heatmap")
    corr_matrix = final_df[CORE_FEATURES + ["FraudLabel"]].corr()
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0F1117")
    ax.set_facecolor("#1E2130")
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5, square=True,
                annot_kws={"size":8, "color":"white"}, ax=ax)
    ax.tick_params(colors="white", labelsize=8)
    ax.set_title("Correlation Matrix", color="white", fontsize=12, pad=15)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "🤖  Model Performance":

    st.title("🤖 Model Performance")
    st.markdown("#### Logistic Regression — Baseline Evaluation")
    st.markdown("---")

    if not MODEL_LOADED:
        st.error("Model files not found.")
        st.stop()

    # Compute metrics — y_test and y_pred are already int arrays
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    ap   = average_precision_score(y_test, y_proba)

    # Metrics row
    cols = st.columns(5)
    metrics_data = [
        (f"{acc*100:.1f}%",  "Accuracy",  "Misleading — ignore",       ""),
        (f"{prec*100:.1f}%", "Precision", "Of flagged, % real fraud",  ""),
        (f"{rec*100:.1f}%",  "Recall",    "Of fraud, % caught",        "good"),
        (f"{f1*100:.1f}%",   "F1-Score",  "Precision-Recall balance",  ""),
        (f"{auc:.3f}",       "AUC-ROC",   "Overall model quality",     "good"),
    ]

    for col, (v, l, s, c) in zip(cols, metrics_data):
        with col:
            st.markdown(f"""<div class="metric-card {c}">
                <p class="metric-value">{v}</p>
                <p class="metric-label">{l}</p>
                <p style="font-size:0.75rem;color:#8892A4;margin:4px 0 0 0">{s}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

        fig, ax = plt.subplots(figsize=(8,6), facecolor="#0F1117")
        ax.set_facecolor("#1E2130")
        labels_cm = np.array([
            [f"TN\n{TN}\nCorrectly cleared", f"FP\n{FP}\nFalse alarm"],
            [f"FN\n{FN}\nMissed fraud",       f"TP\n{TP}\nCaught fraud"]
        ])
        sns.heatmap(cm, annot=labels_cm, fmt="", cmap="Blues",
                    xticklabels=["Predicted\nNot Fraud","Predicted\nFraud"],
                    yticklabels=["Actual\nNot Fraud","Actual\nFraud"],
                    linewidths=2, linecolor="#0F1117",
                    annot_kws={"size":10,"weight":"bold"}, ax=ax)
        ax.tick_params(colors="white", labelsize=9)
        ax.set_title("Confusion Matrix", color="white", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(f"""
        <div class="insight-box warning">
        ⚠️ <b>Missed Fraud (FN): {FN}</b> — providers still billing Medicare undetected.
        </div>
        <div class="insight-box success">
        ✓ <b>Caught Fraud (TP): {TP}</b> — out of {TP+FN} total fraud providers ({rec*100:.1f}% Recall).
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("### Classification Report")
        report_dict = classification_report(
            y_test, y_pred,
            target_names=["Not Fraud","Fraud"],
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report_dict).T.round(3),
                     use_container_width=True)

        st.markdown("### Feature Coefficients")
        coef_df = pd.DataFrame({
            "Feature"    : CORE_FEATURES,
            "Coefficient": model.coef_[0]
        }).sort_values("Coefficient", ascending=True)

        fig, ax = plt.subplots(figsize=(8,6), facecolor="#0F1117")
        ax.set_facecolor("#1E2130")
        colors = ["#E84C4C" if c > 0 else "#4C9BE8"
                  for c in coef_df["Coefficient"]]
        ax.barh(coef_df["Feature"], coef_df["Coefficient"],
                color=colors, edgecolor="none")
        ax.axvline(0, color="#8892A4", linewidth=1)
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[:].set_visible(False)
        ax.set_title("Feature Coefficients", color="white", fontsize=11)
        ax.set_xlabel("Coefficient", color="white")
        red_p  = mpatches.Patch(color="#E84C4C", label="Increases fraud probability")
        blue_p = mpatches.Patch(color="#4C9BE8", label="Decreases fraud probability")
        ax.legend(handles=[red_p, blue_p], facecolor="#1E2130",
                  labelcolor="white", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("### SMOTE vs class_weight='balanced'")
    comp = pd.DataFrame({
        "Metric"                : ["Precision","Recall","F1-Score","AUC-ROC"],
        "SMOTE"                 : ["46.4%","88.1%","60.8%","0.962"],
        "class_weight=balanced" : ["46.1%","88.1%","60.5%","0.961"],
        "Winner"                : ["SMOTE ✓","Tie","SMOTE ✓","SMOTE ✓"],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.markdown("""<div class="insight-box info">
    ℹ️ Both methods produce nearly identical results. <b>SMOTE</b> wins
    marginally and is selected as the final approach.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — ROC & PR CURVES
# ════════════════════════════════════════════════════════════
elif page == "📈  ROC & PR Curves":

    st.title("📈 ROC & Precision-Recall Curves")
    st.markdown("---")

    if not MODEL_LOADED:
        st.error("Model files not found.")
        st.stop()

    fpr, tpr, thr_roc     = roc_curve(y_test, y_proba)
    auc_roc               = roc_auc_score(y_test, y_proba)
    prec_c, rec_c, _      = precision_recall_curve(y_test, y_proba)
    ap_score              = average_precision_score(y_test, y_proba)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ROC Curve")
        fig, ax = plt.subplots(figsize=(9,6), facecolor="#0F1117")
        ax.set_facecolor("#1E2130")
        ax.plot(fpr, tpr, color="#4C9BE8", linewidth=2.5,
                label=f"Logistic Regression (AUC = {auc_roc:.3f})")
        ax.plot([0,1],[0,1], color="#8892A4", linestyle="--",
                linewidth=1, label="Random Guess (AUC = 0.500)")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#4C9BE8")
        idx = np.argmin(np.abs(thr_roc - 0.5))
        ax.scatter(fpr[idx], tpr[idx], color="#E84C4C", s=100, zorder=5,
                   label=f"Threshold=0.5 (Recall={tpr[idx]:.2f})")
        ax.set_xlabel("False Positive Rate", color="white", fontsize=9)
        ax.set_ylabel("True Positive Rate (Recall)", color="white", fontsize=9)
        ax.set_title("ROC Curve", color="white", fontsize=12)
        ax.legend(facecolor="#1E2130", labelcolor="white", fontsize=8)
        ax.tick_params(colors="white")
        ax.spines[:].set_visible(False)
        ax.grid(True, alpha=0.1, color="white")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(f"""<div class="insight-box success">
        ✓ <b>AUC-ROC = {auc_roc:.3f}</b> — Model correctly ranks fraud provider
        as more suspicious than a legitimate one <b>{auc_roc*100:.1f}%</b> of the time.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### Precision-Recall Curve")
        fig, ax = plt.subplots(figsize=(7,5), facecolor="#0F1117")
        ax.set_facecolor("#1E2130")
        ax.plot(rec_c, prec_c, color="#E84C4C", linewidth=2.5,
                label=f"Logistic Regression (AP = {ap_score:.3f})")
        ax.axhline(y=y_test.mean(), color="#8892A4", linestyle="--",
                   linewidth=1, label=f"Baseline (AP = {y_test.mean():.3f})")
        ax.fill_between(rec_c, prec_c, y_test.mean(), alpha=0.1, color="#E84C4C")
        ax.set_xlabel("Recall", color="white", fontsize=9)
        ax.set_ylabel("Precision", color="white", fontsize=9)
        ax.set_title("Precision-Recall Curve", color="white", fontsize=12)
        ax.legend(facecolor="#1E2130", labelcolor="white", fontsize=8)
        ax.tick_params(colors="white")
        ax.spines[:].set_visible(False)
        ax.grid(True, alpha=0.1, color="white")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(f"""<div class="insight-box success">
        ✓ <b>Average Precision = {ap_score:.3f}</b> — Our model is
        <b>{ap_score/y_test.mean():.1f}x better</b> than random guessing.
        PR curve is more informative than ROC for imbalanced datasets.
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 5 — THRESHOLD TUNER
# ════════════════════════════════════════════════════════════
elif page == "🎚️  Threshold Tuner":

    st.title("🎚️ Interactive Threshold Tuner")
    st.markdown("#### Adjust the decision threshold and see live impact")
    st.markdown("---")

    if not MODEL_LOADED:
        st.error("Model files not found.")
        st.stop()

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10, max_value=0.90,
        value=0.50, step=0.01,
        help="Lower = catch more fraud. Higher = fewer false alarms."
    )

    y_pred_t   = (y_proba >= threshold).astype(int)
    p_t        = precision_score(y_test, y_pred_t, zero_division=0)
    r_t        = recall_score(y_test, y_pred_t)
    f_t        = f1_score(y_test, y_pred_t, zero_division=0)
    cm_t       = confusion_matrix(y_test, y_pred_t)
    TN_t, FP_t, FN_t, TP_t = cm_t[0,0], cm_t[0,1], cm_t[1,0], cm_t[1,1]
    total_fraud = int((y_test == 1).sum())

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card good">
            <p class="metric-value">{r_t*100:.1f}%</p>
            <p class="metric-label">Recall</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value">{p_t*100:.1f}%</p>
            <p class="metric-label">Precision</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <p class="metric-value">{f_t*100:.1f}%</p>
            <p class="metric-label">F1-Score</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card fraud">
            <p class="metric-value">{FN_t}</p>
            <p class="metric-label">Missed Fraud</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Precision & Recall vs Threshold")
        thresholds_range = np.arange(0.1, 0.9, 0.01)
        precisions_r, recalls_r, f1s_r = [], [], []
        for t in thresholds_range:
            yp = (y_proba >= t).astype(int)
            precisions_r.append(precision_score(y_test, yp, zero_division=0))
            recalls_r.append(recall_score(y_test, yp))
            f1s_r.append(f1_score(y_test, yp, zero_division=0))

        fig, ax = plt.subplots(figsize=(10,6), facecolor="#0F1117")
        ax.set_facecolor("#1E2130")
        ax.plot(thresholds_range, recalls_r,    color="#E84C4C", linewidth=2, label="Recall")
        ax.plot(thresholds_range, precisions_r, color="#4C9BE8", linewidth=2, label="Precision")
        ax.plot(thresholds_range, f1s_r,        color="#4CE89B", linewidth=2,
                linestyle="--", label="F1-Score")
        ax.axvline(threshold, color="orange", linestyle="--",
                   linewidth=2, label=f"Current: {threshold}")
        ax.scatter([threshold],[r_t], color="#E84C4C", s=80, zorder=5)
        ax.scatter([threshold],[p_t], color="#4C9BE8", s=80, zorder=5)
        ax.set_xlabel("Threshold", color="white")
        ax.set_ylabel("Score", color="white")
        ax.set_title("Precision, Recall & F1 vs Threshold", color="white", fontsize=11)
        ax.legend(facecolor="#1E2130", labelcolor="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.spines[:].set_visible(False)
        ax.grid(True, alpha=0.1, color="white")
        ax.set_xlim(0.1, 0.9)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        st.markdown("### Live Confusion Matrix")
        fig, ax = plt.subplots(figsize=(10,6), facecolor="#0F1117")
        ax.set_facecolor("#1E2130")
        labels_cm = np.array([
            [f"TN\n{TN_t}\nCorrectly cleared", f"FP\n{FP_t}\nFalse alarm"],
            [f"FN\n{FN_t}\nMissed fraud",       f"TP\n{TP_t}\nCaught fraud"]
        ])
        sns.heatmap(cm_t, annot=labels_cm, fmt="", cmap="Blues",
                    xticklabels=["Predicted\nNot Fraud","Predicted\nFraud"],
                    yticklabels=["Actual\nNot Fraud","Actual\nFraud"],
                    linewidths=2, linecolor="#0F1117",
                    annot_kws={"size":10,"weight":"bold"}, ax=ax)
        ax.tick_params(colors="white", labelsize=9)
        ax.set_title(f"Threshold = {threshold}", color="white", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("### 💼 Business Interpretation")
        st.markdown(f"""<div class="insight-box info">
        At threshold <b>{threshold}</b>:<br><br>
        ✓ Catches <b>{TP_t} of {total_fraud}</b> fraud providers ({r_t*100:.1f}% Recall)<br>
        ✗ Misses <b>{FN_t}</b> fraud providers still billing Medicare<br>
        ⚠️ Generates <b>{FP_t}</b> false alarms (innocent providers flagged)<br><br>
        For every <b>100 providers investigated</b>,
        approximately <b>{p_t*100:.0f} are genuinely fraudulent</b>.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Threshold Comparison Table")
    table_data = []
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        yp = (y_proba >= t).astype(int)
        caught = int(yp[y_test == 1].sum())
        table_data.append({
            "Threshold"   : t,
            "Precision"   : f"{precision_score(y_test, yp, zero_division=0)*100:.1f}%",
            "Recall"      : f"{recall_score(y_test, yp)*100:.1f}%",
            "F1-Score"    : f"{f1_score(y_test, yp, zero_division=0)*100:.1f}%",
            "Caught Fraud": f"{caught}/{total_fraud}",
            "Missed"      : int(total_fraud - caught),
        })
    st.dataframe(pd.DataFrame(table_data),
                 use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# PAGE 6 — LIVE PREDICTION
# ════════════════════════════════════════════════════════════
elif page == "🔮  Live Prediction":

    st.title("🔮 Live Fraud Prediction")
    st.markdown("#### Enter provider details to predict fraud probability")
    st.markdown("---")

    if not MODEL_LOADED:
        st.error("Model files not found.")
        st.stop()

    st.markdown("### Provider Details")
    col1, col2 = st.columns(2)

    with col1:
        total_claims     = st.number_input("Total Claims Submitted",    min_value=1,   max_value=10000,   value=100)
        total_reimbursed = st.number_input("Total Reimbursed ($)",      min_value=0,   max_value=10000000,value=50000)
        avg_reimbursed   = st.number_input("Avg Reimbursed per Claim ($)", min_value=0,max_value=100000,  value=500)
        avg_claim_dur    = st.number_input("Avg Claim Duration (days)", min_value=0.0, max_value=60.0,    value=2.0, step=0.5)
        avg_days_hosp    = st.number_input("Avg Days in Hospital",      min_value=0.0, max_value=60.0,    value=0.0, step=0.5)

    with col2:
        unique_patients  = st.number_input("Unique Patients Seen",         min_value=1,   max_value=5000, value=30)
        unique_phys      = st.number_input("Unique Attending Physicians",   min_value=1,   max_value=1000, value=5)
        same_attend_rate = st.slider("Same Attend/Oper Rate",  0.0, 1.0, 0.1, 0.01)
        avg_chronic      = st.number_input("Avg Chronic Conditions/Patient",min_value=0.0,max_value=11.0, value=4.5, step=0.1)
        inpatient_ratio  = st.slider("Inpatient Claim Ratio",  0.0, 1.0, 0.0, 0.01)

    st.markdown("---")
    predict_col, _ = st.columns([1, 2])
    with predict_col:
        predict_btn = st.button("🔍 Predict Fraud Probability",
                                use_container_width=True, type="primary")

    if predict_btn:
        input_data = np.array([[
            total_claims, total_reimbursed, avg_reimbursed,
            avg_claim_dur, avg_days_hosp, unique_patients,
            unique_phys, same_attend_rate, avg_chronic, inpatient_ratio
        ]])
        input_scaled = scaler.transform(input_data)
        fraud_prob   = float(model.predict_proba(input_scaled)[0][1])

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        if fraud_prob >= 0.7:
            risk_level, risk_color, risk_icon = "HIGH RISK",   "#E84C4C", "🚨"
        elif fraud_prob >= 0.4:
            risk_level, risk_color, risk_icon = "MEDIUM RISK", "#E8A84C", "⚠️"
        else:
            risk_level, risk_color, risk_icon = "LOW RISK",    "#4CE89B", "✅"

        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1E2130,#252A3D);
                        border:2px solid {risk_color};border-radius:16px;
                        padding:30px;text-align:center;">
                <p style="font-size:3rem;margin:0">{risk_icon}</p>
                <p style="font-size:2.5rem;font-weight:700;
                           color:{risk_color};margin:8px 0">{fraud_prob*100:.1f}%</p>
                <p style="font-size:1rem;color:{risk_color};
                           font-weight:600;margin:0">{risk_level}</p>
                <p style="color:#8892A4;font-size:0.85rem;
                           margin:8px 0 0 0">Fraud Probability</p>
            </div>""", unsafe_allow_html=True)

        with col_res2:
            st.markdown("#### Feature Contributions")
            contributions = input_scaled[0] * model.coef_[0]
            contrib_df = pd.DataFrame({
                "Feature"     : CORE_FEATURES,
                "Contribution": contributions
            }).sort_values("Contribution", key=abs, ascending=True)

            fig, ax = plt.subplots(figsize=(8,5), facecolor="#0F1117")
            ax.set_facecolor("#1E2130")
            colors = ["#E84C4C" if c > 0 else "#4C9BE8"
                      for c in contrib_df["Contribution"]]
            ax.barh(contrib_df["Feature"], contrib_df["Contribution"],
                    color=colors, edgecolor="none")
            ax.axvline(0, color="#8892A4", linewidth=1)
            ax.tick_params(colors="white", labelsize=9)
            ax.spines[:].set_visible(False)
            ax.set_title("Feature Contributions to Fraud Score",
                         color="white", fontsize=10)
            ax.set_xlabel("Contribution (red = increases fraud score)",
                          color="white", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Plain English
            top_fraud = contrib_df[contrib_df["Contribution"] > 0]["Feature"].tolist()[-3:]
            top_legit = contrib_df[contrib_df["Contribution"] < 0]["Feature"].tolist()[-3:]
            st.markdown("#### 📝 Plain English Explanation")
            if top_fraud:
                st.markdown(f"""<div class="insight-box warning">
                🚨 <b>Fraud indicators:</b> {", ".join(top_fraud)} are elevated
                compared to typical providers.
                </div>""", unsafe_allow_html=True)
            if top_legit:
                st.markdown(f"""<div class="insight-box success">
                ✅ <b>Legitimate indicators:</b> {", ".join(top_legit)} are within normal range.
                </div>""", unsafe_allow_html=True)