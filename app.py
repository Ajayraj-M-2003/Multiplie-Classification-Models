import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Credit Default Classifier | BITS Pilani ML Assignment 2",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished, academic look
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .badge-bits {
        display: inline-block;
        background: linear-gradient(135deg, #1E3A8A, #2563EB);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model Catalog & File Paths
# ---------------------------------------------------------
MODEL_DIR = "model"
MODEL_FILES = {
    "Decision Tree": os.path.join(MODEL_DIR, "decision_tree.joblib"),
    "Support Vector Machine": os.path.join(MODEL_DIR, "svm.joblib"),
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression.joblib"),
    "KNN": os.path.join(MODEL_DIR, "knn.joblib"),
    "Naive Bayes": os.path.join(MODEL_DIR, "naive_bayes.joblib")
}

EXPECTED_FEATURES = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# ---------------------------------------------------------
# Helper Functions with Streamlit Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load all pre-trained pipeline models from disk."""
    loaded_models = {}
    for name, filepath in MODEL_FILES.items():
        if os.path.exists(filepath):
            loaded_models[name] = joblib.load(filepath)
    return loaded_models


@st.cache_data(show_spinner=False)
def load_default_test_data():
    """Load bundled test_data.csv."""
    default_path = "test_data.csv"
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None


def extract_features_and_target(df: pd.DataFrame):
    """
    Identifies the target column and extracts feature matrix X and ground truth y.
    Handles multiple naming variants for robustness.
    """
    possible_targets = [
        'default', 'default payment next month', 'default_payment_next_month',
        'target', 'Default', 'TARGET'
    ]
    target_col = None
    for cand in possible_targets:
        if cand in df.columns:
            target_col = cand
            break

    if target_col is None:
        return None, None, f"Could not find target column. Expected one of: {possible_targets}"

    # Verify input features
    missing_feats = [col for col in EXPECTED_FEATURES if col not in df.columns]
    if missing_feats:
        return None, None, f"Missing required features ({len(missing_feats)}): {missing_feats[:5]}..."

    X = df[EXPECTED_FEATURES]
    y = df[target_col].astype(int)
    return X, y, None


def compute_metrics(y_true, y_pred, y_scores=None):
    """Calculates all 6 mandatory evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    auc = None
    if y_scores is not None:
        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            auc = None

    return {
        "Accuracy": acc,
        "AUC": auc if auc is not None else 0.0,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc
    }


def predict_model(pipeline, X):
    """Generates predictions and ranking scores (for ROC-AUC)."""
    y_pred = pipeline.predict(X)
    clf = pipeline.named_steps.get('classifier', pipeline)
    if hasattr(clf, 'predict_proba'):
        y_scores = pipeline.predict_proba(X)[:, 1]
    elif hasattr(clf, 'decision_function'):
        y_scores = pipeline.decision_function(X)
    else:
        y_scores = y_pred
    return y_pred, y_scores


# ---------------------------------------------------------
# Sidebar: Navigation & Data Ingestion
# ---------------------------------------------------------
models_dict = load_all_models()

with st.sidebar:
    st.markdown('<div class="badge-bits">BITS Pilani • WILP M.Tech AIML</div>', unsafe_allow_html=True)
    st.title("💳 Navigation & Data")

    st.markdown("### 1. Model Selection")
    view_options = ["Model Comparison (All 5)"] + list(MODEL_FILES.keys())
    selected_view = st.selectbox(
        "Choose Model or Comparison View:",
        options=view_options,
        index=0,
        help="Select an individual classifier to inspect detailed metrics, confusion matrix, and ROC curve, or choose Comparison to see all 5 models."
    )

    st.markdown("---")
    st.markdown("### 2. Test Data Upload (CSV)")
    st.info("💡 As per assignment guidelines, upload test data only. The app computes metrics without retraining.")

    uploaded_file = st.file_uploader(
        "Upload Test CSV",
        type=["csv"],
        help="Upload a test CSV containing the 23 features and target column 'default'."
    )

    data_source_name = ""
    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
            data_source_name = f"Uploaded File: `{uploaded_file.name}` ({len(test_df):,} rows)"
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            test_df = None
    else:
        test_df = load_default_test_data()
        if test_df is not None:
            data_source_name = f"Default Bundled `test_data.csv` ({len(test_df):,} rows)"
        else:
            data_source_name = "None"

    st.markdown(f"**Current Data**: {data_source_name}")

    st.markdown("---")
    st.markdown("""
    **Assignment Details:**
    - **Course**: Machine Learning (M.Tech AIML)
    - **Assignment**: Assignment 2
    - **Dataset**: UCI Credit Card Default (23 features)
    - **Target**: Default Payment Next Month (0/1)
    """)

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">Credit Card Default Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Comparative Evaluation of 5 Machine Learning Classifiers with Real-Time Inference on Test Data</div>',
    unsafe_allow_html=True
)

if test_df is None:
    st.error("⚠️ No test data loaded. Please upload a test CSV from the sidebar or ensure `test_data.csv` exists in the repository root.")
    st.stop()

X_test, y_test, err_msg = extract_features_and_target(test_df)
if err_msg:
    st.error(f"⚠️ Data Schema Error: {err_msg}")
    st.stop()

# ---------------------------------------------------------
# Data Summary Banner
# ---------------------------------------------------------
with st.expander("🔍 Test Dataset Overview & Feature Inspector", expanded=False):
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Test Instances", f"{len(test_df):,}")
    col_b.metric("Total Input Features", f"{len(EXPECTED_FEATURES)}")
    neg_count = int((y_test == 0).sum())
    pos_count = int((y_test == 1).sum())
    col_c.metric("Class 0 (Non-Default)", f"{neg_count:,} ({neg_count/len(y_test)*100:.1f}%)")
    col_d.metric("Class 1 (Default)", f"{pos_count:,} ({pos_count/len(y_test)*100:.1f}%)")
    
    st.dataframe(test_df.head(10), use_container_width=True)

# ---------------------------------------------------------
# View 1: All Models Comparison Table & Charts
# ---------------------------------------------------------
if selected_view == "Model Comparison (All 5)":
    st.subheader("📊 Model Performance Comparison Table")
    st.write("Evaluating all 5 pre-trained models on the active test dataset:")

    comparison_records = []
    roc_curves_data = {}

    for name in MODEL_FILES.keys():
        if name in models_dict:
            pipeline = models_dict[name]
            y_pred, y_scores = predict_model(pipeline, X_test)
            m = compute_metrics(y_test, y_pred, y_scores)
            comparison_records.append({
                "ML Model Name": name,
                "Accuracy": m["Accuracy"],
                "AUC": m["AUC"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1": m["F1"],
                "MCC": m["MCC"]
            })
            if y_scores is not None:
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_curves_data[name] = (fpr, tpr, m["AUC"])

    comp_df = pd.DataFrame(comparison_records)

    # Highlight maximum for each metric column
    styled_df = comp_df.style.format({
        "Accuracy": "{:.4f}",
        "AUC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "MCC": "{:.4f}"
    }).highlight_max(
        subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        color="#D1FAE5"
    )

    st.dataframe(styled_df, use_container_width=True)

    # Visual Comparison Charts
    st.markdown("### 📈 Visual Benchmark")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Metric Comparison Across Models (Bar Chart)**")
        plot_df = comp_df.melt(
            id_vars=["ML Model Name"],
            value_vars=["Accuracy", "AUC", "F1", "MCC"],
            var_name="Metric",
            value_name="Score"
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=plot_df, x="ML Model Name", y="Score", hue="Metric", palette="tab10", ax=ax)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_ylabel("Metric Score")
        ax.set_title("Key Performance Metrics Comparison", fontsize=12, fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

    with chart_col2:
        st.markdown("**Combined Receiver Operating Characteristic (ROC) Curves**")
        fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
        for model_name, (fpr, tpr, auc_val) in roc_curves_data.items():
            ax_roc.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.4f})", linewidth=2)
        ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance Baseline (AUC = 0.50)")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
        ax_roc.set_ylabel("True Positive Rate (Recall)")
        ax_roc.set_title("ROC Curves for All Classifiers", fontsize=12, fontweight="bold")
        ax_roc.legend(loc="lower right", fontsize=9)
        ax_roc.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_roc)

    # Performance Observations Summary matching Section 3 - Step 5
    st.markdown("---")
    st.subheader("📝 Observations on Model Performance")
    
    obs_col1, obs_col2 = st.columns(2)
    with obs_col1:
        st.markdown("""
        - **Decision Tree**: Highest overall **ROC-AUC (0.7462)**, **F1-Score (0.4657)**, and **MCC (0.3909)**. By learning non-linear threshold rules on repayment history (`PAY_0` to `PAY_6`), it captures risk delinquency earlier than linear models.
        - **Support Vector Machine (RBF)**: Competitive **Accuracy (0.8160)** and solid **Precision (0.6652)**. RBF kernel models non-linear decision boundaries well, though sensitive to scaling.
        - **Logistic Regression**: Reliable baseline with strong **Accuracy (0.8077)** and highest **Precision (0.6868)**, but lower recall due to class imbalance.
        """)
    with obs_col2:
        st.markdown("""
        - **K-Nearest Neighbors (KNN)**: Achieves **0.7980 Accuracy** and **0.7085 AUC**. Distance weighting helps distinguish local density, though inference cost scales with dataset size.
        - **Naive Bayes (Gaussian)**: Highest **Recall (0.5539)** among all models, identifying more true defaulters at the expense of lower precision due to feature correlation violations.
        - **🏆 Overall Winner**: **Decision Tree Classifier** ranks #1 overall due to its balanced performance across AUC, F1, and MCC, as well as immediate interpretability.
        """)

# ---------------------------------------------------------
# View 2: Individual Model Deep Dive
# ---------------------------------------------------------
else:
    model_name = selected_view
    st.subheader(f"🎯 Deep-Dive Evaluation: {model_name}")

    if model_name not in models_dict:
        st.error(f"Model file for {model_name} not found in `{MODEL_DIR}/`.")
        st.stop()

    pipeline = models_dict[model_name]
    y_pred, y_scores = predict_model(pipeline, X_test)
    metrics = compute_metrics(y_test, y_pred, y_scores)

    # 1. KPI Metric Cards (Section 5c)
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Accuracy</div><div class="metric-value">{metrics["Accuracy"]:.4f}</div></div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">AUC Score</div><div class="metric-value">{metrics["AUC"]:.4f}</div></div>', unsafe_allow_html=True)
    with m_col3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Precision</div><div class="metric-value">{metrics["Precision"]:.4f}</div></div>', unsafe_allow_html=True)
    with m_col4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Recall</div><div class="metric-value">{metrics["Recall"]:.4f}</div></div>', unsafe_allow_html=True)
    with m_col5:
        st.markdown(f'<div class="metric-card"><div class="metric-title">F1 Score</div><div class="metric-value">{metrics["F1"]:.4f}</div></div>', unsafe_allow_html=True)
    with m_col6:
        st.markdown(f'<div class="metric-card"><div class="metric-title">MCC</div><div class="metric-value">{metrics["MCC"]:.4f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Confusion Matrix & Classification Report (Section 5d)
    vis_col1, vis_col2 = st.columns([1, 1])

    with vis_col1:
        st.markdown("### 🔲 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        
        # Detailed annotations with count & percentage
        group_counts = [f"{value:,}" for value in cm.flatten()]
        group_percentages = [f"{value:.1%}" for value in cm.flatten() / np.sum(cm)]
        labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        
        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax_cm,
            xticklabels=["Predicted No Default (0)", "Predicted Default (1)"],
            yticklabels=["Actual No Default (0)", "Actual Default (1)"]
        )
        ax_cm.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold")
        ax_cm.set_ylabel("True Label")
        ax_cm.set_xlabel("Predicted Label")
        plt.tight_layout()
        st.pyplot(fig_cm)

    with vis_col2:
        st.markdown("### 📋 Classification Report")
        report_dict = classification_report(
            y_test, y_pred,
            target_names=["Class 0 (Non-Default)", "Class 1 (Default)"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df.round(4)
        st.dataframe(
            report_df.style.highlight_max(subset=["precision", "recall", "f1-score"], color="#E0F2FE"),
            use_container_width=True
        )

        # ROC Curve for selected model
        if y_scores is not None:
            st.markdown("### 📉 ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            fig_single_roc, ax_s_roc = plt.subplots(figsize=(6, 3.5))
            ax_s_roc.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC (AUC = {metrics['AUC']:.4f})")
            ax_s_roc.plot([0, 1], [0, 1], color="grey", linestyle="--")
            ax_s_roc.fill_between(fpr, tpr, alpha=0.15, color="#2563EB")
            ax_s_roc.set_xlim([0.0, 1.0])
            ax_s_roc.set_ylim([0.0, 1.05])
            ax_s_roc.set_xlabel("False Positive Rate")
            ax_s_roc.set_ylabel("True Positive Rate")
            ax_s_roc.legend(loc="lower right")
            ax_s_roc.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_single_roc)

    # 3. Sample Inference Table
    st.markdown("### 🧪 Sample Predictions on Test Data")
    sample_preview = X_test.head(15).copy()
    sample_preview["Actual Target"] = y_test.head(15).values
    sample_preview["Predicted Class"] = y_pred[:15]
    if y_scores is not None:
        sample_preview["Model Score/Probability"] = np.round(y_scores[:15], 4)
    st.dataframe(sample_preview, use_container_width=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94A3B8; font-size: 0.85rem;">
        BITS Pilani • Work Integrated Learning Programmes (WILP) • M.Tech AIML — Machine Learning Assignment 2<br>
        Models serialized via scikit-learn & joblib • Built with Streamlit Community Cloud
    </div>
    """,
    unsafe_allow_html=True
)

