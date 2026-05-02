import os
import io
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Churn Intelligence · XGBoost",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: #0a0e1a;
        color: #e2e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1224 0%, #111827 100%);
        border-right: 1px solid #1e2940;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.93rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #141c2f 0%, #1a2540 100%);
        border: 1px solid #1e2e50;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(56,139,253,.18);
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.1rem;
        font-weight: 700;
        color: #60a5fa;
        margin: 0;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748b;
        margin-top: 4px;
    }
    .kpi-delta {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Section titles */
    .section-header {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f1f5f9;
        border-left: 3px solid #3b82f6;
        padding-left: 12px;
        margin-bottom: 18px;
    }

    /* Probability badge */
    .prob-high {
        background: rgba(239,68,68,.15);
        border: 1px solid #ef4444;
        color: #fca5a5;
        border-radius: 8px;
        padding: 8px 14px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        text-align: center;
    }
    .prob-low {
        background: rgba(34,197,94,.12);
        border: 1px solid #22c55e;
        color: #86efac;
        border-radius: 8px;
        padding: 8px 14px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        text-align: center;
    }
    .prob-med {
        background: rgba(245,158,11,.12);
        border: 1px solid #f59e0b;
        color: #fcd34d;
        border-radius: 8px;
        padding: 8px 14px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        text-align: center;
    }

    /* Info box */
    .info-box {
        background: rgba(59,130,246,.08);
        border: 1px solid #1d4ed8;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.88rem;
        color: #93c5fd;
        margin-bottom: 12px;
    }

    /* Divider */
    hr { border-color: #1e2940; }

    /* Streamlit overrides */
    .stSlider > div { padding: 0; }
    div[data-testid="metric-container"] {
        background: #141c2f;
        border: 1px solid #1e2e50;
        border-radius: 10px;
        padding: 16px 20px;
    }
    .stDataFrame { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)
MODEL_PATH = os.path.join("model", "xgb_churn_model.pkl")
THRESH_PATH = os.path.join("model", "xgb_churn_threshold.pkl")
DEFAULT_THRESHOLD = 0.38

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Syne, sans-serif", color="#cbd5e1", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#3b82f6", "#06b6d4", "#8b5cf6", "#f59e0b", "#10b981", "#ef4444"],
)

FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
    "MonthlyCharges", "TotalCharges", "HasInternet",
    "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    "PhoneLineStatus_Multiple Lines",
    "PhoneLineStatus_No Phone Service",
    "PhoneLineStatus_Single Line",
]

BINARY_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "PaperlessBilling", "HasInternet",
]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

OHE_GROUPS = {
    "InternetService": ["InternetService_DSL", "InternetService_Fiber optic", "InternetService_No"],
    "Contract":        ["Contract_Month-to-month", "Contract_One year", "Contract_Two year"],
    "PaymentMethod":   [
        "PaymentMethod_Bank transfer (automatic)",
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check",
    ],
    "PhoneLineStatus": [
        "PhoneLineStatus_Multiple Lines",
        "PhoneLineStatus_No Phone Service",
        "PhoneLineStatus_Single Line",
    ],
}

# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_threshold():
    """Load XGBoost model and optimal threshold from disk."""
    model, threshold, error = None, DEFAULT_THRESHOLD, None
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        error = f"Model tidak ditemukan di `{MODEL_PATH}`: {e}"
    try:
        threshold = float(joblib.load(THRESH_PATH))
    except Exception:
        threshold = DEFAULT_THRESHOLD
    return model, threshold, error


def get_feature_names(model) -> list[str]:
    """Get feature names from model or fallback to spec."""
    try:
        return list(model.feature_names_in_)
    except AttributeError:
        pass
    try:
        return model.get_booster().feature_names
    except Exception:
        return FEATURE_COLUMNS


def get_feature_importances(model, feature_names: list[str]) -> pd.DataFrame:
    """Return tidy DataFrame of feature importances."""
    rows = []
    try:
        booster = model.get_booster()
        for score_type in ("weight", "gain", "cover"):
            scores = booster.get_score(importance_type=score_type)
            for feat, val in scores.items():
                rows.append({"feature": feat, "type": score_type, "importance": val})
        df = pd.DataFrame(rows)
        return df
    except Exception:
        pass
    try:
        imp = model.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "type": "weight",
            "importance": imp,
        })
        return df
    except Exception:
        return pd.DataFrame(columns=["feature", "type", "importance"])

def load_sample_data(n: int = 200, feature_names: list[str] | None = None) -> pd.DataFrame:
    """Generate a synthetic DataFrame matching model feature columns."""
    cols = feature_names or FEATURE_COLUMNS
    rng = np.random.default_rng(42)
    data = {}
    for col in cols:
        if col in BINARY_FEATURES:
            data[col] = rng.integers(0, 2, n)
        elif col in NUMERIC_FEATURES:
            if col == "tenure":
                data[col] = rng.integers(0, 72, n).astype(float)
            elif col == "MonthlyCharges":
                data[col] = rng.uniform(18, 120, n)
            else:
                data[col] = rng.uniform(0, 8000, n)
        else:  # OHE booleans
            data[col] = rng.integers(0, 2, n)
    df = pd.DataFrame(data)
    # Fix OHE: ensure exactly one hot per group
    for group_cols in OHE_GROUPS.values():
        present = [c for c in group_cols if c in df.columns]
        if present:
            choices = rng.integers(0, len(present), n)
            for i, c in enumerate(present):
                df[c] = (choices == i).astype(int)
    return df

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Compute all classification metrics at given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
        "cm": cm,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "y_pred": y_pred,
    }


def fig_confusion_matrix(cm: np.ndarray) -> go.Figure:
    labels = ["No Churn (0)", "Churn (1)"]
    text = [[str(cm[i][j]) for j in range(2)] for i in range(2)]
    total = cm.sum()
    pct   = [[f"{cm[i][j]/total*100:.1f}%" for j in range(2)] for i in range(2)]
    hover = [[f"{text[i][j]} samples<br>{pct[i][j]}" for j in range(2)] for i in range(2)]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels, y=labels,
        colorscale=[[0, "#0f1729"], [0.5, "#1d4ed8"], [1, "#3b82f6"]],
        showscale=False,
        text=[[f"<b>{text[i][j]}</b><br><span style='font-size:11px;color:#94a3b8'>{pct[i][j]}</span>"
               for j in range(2)] for i in range(2)],
        hovertext=hover,
        texttemplate="%{text}",
        hovertemplate="%{hovertext}<extra></extra>",
        textfont=dict(size=18),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(title="Predicted", side="bottom", showgrid=False),
        yaxis=dict(title="Actual", autorange="reversed", showgrid=False),
        height=320,
        title=dict(text="Confusion Matrix", x=0.5, xanchor="center", font=dict(size=14)),
    )
    return fig


def fig_roc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC Curve", "Precision-Recall Curve"))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={auc:.3f}", line=dict(color="#3b82f6", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="#334155"), showlegend=False), row=1, col=1)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = float(np.mean(prec))
    fig.add_trace(go.Scatter(x=rec, y=prec, name=f"Avg Prec={ap:.3f}", line=dict(color="#06b6d4", width=2)), row=1, col=2)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=340,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e2940", title_font=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor="#1e2940", title_font=dict(size=11))
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    return fig


def fig_feature_importance(df_imp: pd.DataFrame, imp_type: str, top_n: int, ascending: bool) -> go.Figure:
    df = df_imp[df_imp["type"] == imp_type].copy()
    df = df.sort_values("importance", ascending=ascending).tail(top_n)
    colors = px.colors.sample_colorscale("Blues", np.linspace(0.3, 1.0, len(df)))
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(300, top_n * 28),
        xaxis=dict(title=f"Importance ({imp_type})", showgrid=True, gridcolor="#1e2940"),
        yaxis=dict(showgrid=False),
        title=dict(text=f"Feature Importance — {imp_type.capitalize()}", x=0.5, xanchor="center"),
    )
    return fig


def fig_threshold_curve(y_true: np.ndarray, y_prob: np.ndarray, active_thresh: float) -> go.Figure:
    thresholds = np.arange(0.10, 0.91, 0.01)
    metrics = {"F1": [], "Precision": [], "Recall": []}
    for t in thresholds:
        y_p = (y_prob >= t).astype(int)
        metrics["F1"].append(f1_score(y_true, y_p, zero_division=0))
        metrics["Precision"].append(precision_score(y_true, y_p, zero_division=0))
        metrics["Recall"].append(recall_score(y_true, y_p, zero_division=0))

    fig = go.Figure()
    colors = {"F1": "#3b82f6", "Precision": "#06b6d4", "Recall": "#f59e0b"}
    for name, vals in metrics.items():
        fig.add_trace(go.Scatter(x=thresholds, y=vals, name=name,
                                 line=dict(color=colors[name], width=2)))

    idx = np.argmin(np.abs(thresholds - active_thresh))
    fig.add_trace(go.Scatter(
        x=[active_thresh], y=[metrics["F1"][idx]],
        mode="markers", name=f"Active ({active_thresh:.2f})",
        marker=dict(color="#ef4444", size=12, symbol="diamond"),
    ))
    fig.add_vline(x=active_thresh, line_dash="dash", line_color="#ef4444", opacity=0.5)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=360,
        xaxis=dict(title="Threshold", showgrid=True, gridcolor="#1e2940"),
        yaxis=dict(title="Score", showgrid=True, gridcolor="#1e2940", range=[0, 1]),
        title=dict(text="Threshold vs Metrics", x=0.5, xanchor="center"),
    )
    return fig


def fig_prediction_dist(y_pred: np.ndarray) -> go.Figure:
    vals, cnts = np.unique(y_pred, return_counts=True)
    labels = ["No Churn" if v == 0 else "Churn" for v in vals]
    fig = go.Figure(go.Pie(
        labels=labels, values=cnts,
        hole=0.55,
        marker=dict(colors=["#22c55e", "#ef4444"]),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} records<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                      title=dict(text="Predicted Distribution", x=0.5, xanchor="center"))
    return fig


# ── KPI Card HTML ─────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, delta: str = "") -> str:
    return f"""
    <div class="kpi-card">
        <p class="kpi-value">{value}</p>
        <p class="kpi-label">{label}</p>
        {"<p class='kpi-delta'>" + delta + "</p>" if delta else ""}
    </div>
    """


def align_columns(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Add missing columns as 0, drop extra; return in correct order."""
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


def render_sidebar(threshold_default: float) -> tuple[str, float]:
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center;padding:20px 0 10px'>
                <span style='font-size:2.2rem'>⚡</span>
                <h2 style='margin:4px 0 2px;font-size:1.25rem;font-weight:800;color:#f1f5f9'>
                    Churn Intelligence
                </h2>
                <p style='font-size:0.72rem;letter-spacing:.12em;text-transform:uppercase;
                          color:#475569;margin:0'>XGBoost · Telco · v1.0</p>
            </div>
            <hr style='border-color:#1e2940;margin:0 0 18px'>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigate",
            [
                "📊  Dashboard Overview",
                "📈  Feature Importance",
                "🎯  Threshold Tuning",
                "🔮  Single Prediction",
                "📦  Batch Prediction",
                "ℹ️   Model Info",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#1e2940;margin:18px 0'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;"
            "color:#475569;margin-bottom:8px'>⚙️ Global Threshold</p>",
            unsafe_allow_html=True,
        )
        global_thresh = st.slider(
            "threshold_global",
            0.10, 0.90, float(threshold_default), 0.01,
            label_visibility="collapsed",
            help="Threshold aktif untuk semua halaman. Ubah untuk melihat dampak real-time.",
        )
        st.markdown(
            f"<p style='font-family:JetBrains Mono,monospace;color:#3b82f6;"
            f"font-size:1.1rem;text-align:center;margin:0'>{global_thresh:.2f}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='border-color:#1e2940;margin:18px 0 10px'>", unsafe_allow_html=True)
        st.caption("Built with Streamlit + XGBoost")

    return page, global_thresh

def page_dashboard(model, feature_names: list[str], threshold: float):
    st.markdown("<div class='section-header'>Dashboard Overview</div>", unsafe_allow_html=True)

    st.markdown("<div class='info-box'>📋 Upload test CSV (dengan kolom <code>Churn</code>) untuk evaluasi real. Jika tidak, digunakan data sampel sintetis.</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv", key="dash_upload")

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            if "Churn" in df_raw.columns:
                y_true = df_raw["Churn"].values.astype(int)
                X_raw = df_raw.drop(columns=["Churn"])
            else:
                st.error("Kolom `Churn` tidak ditemukan.")
                return
        except Exception as e:
            st.error(f"Error membaca file: {e}")
            return
    else:
        df_synth = load_sample_data(400, feature_names)
        rng = np.random.default_rng(7)
        y_true = rng.choice([0, 1], size=400, p=[0.73, 0.27])
        X_raw = df_synth

    try:
        X_aligned = align_columns(X_raw.copy(), feature_names)
        y_prob = model.predict_proba(X_aligned)[:, 1]
    except Exception as e:
        st.error(f"Inferensi gagal: {e}")
        return

    m = compute_metrics(y_true, y_prob, threshold)

    cols = st.columns(6)
    kpis = [
        ("Accuracy",  f"{m['accuracy']:.3f}",  "Overall correct"),
        ("Precision", f"{m['precision']:.3f}", "TP / (TP+FP)"),
        ("Recall",    f"{m['recall']:.3f}",    "TP / (TP+FN)"),
        ("F1-Score",  f"{m['f1']:.3f}",        "Harmonic mean"),
        ("ROC-AUC",   f"{m['roc_auc']:.3f}",   "Discrimination"),
        ("Threshold", f"{threshold:.2f}",       "Active cutoff"),
    ]
    for col, (label, val, delta) in zip(cols, kpis):
        col.markdown(kpi_card(label, val, delta), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.8])
    with col1:
        st.plotly_chart(fig_confusion_matrix(m["cm"]), use_container_width=True)
        detail_cols = st.columns(4)
        for dc, (lbl, val, color) in zip(detail_cols, [
            ("TP", m["tp"], "#22c55e"), ("TN", m["tn"], "#3b82f6"),
            ("FP", m["fp"], "#f59e0b"), ("FN", m["fn"], "#ef4444"),
        ]):
            dc.markdown(
                f"<div style='text-align:center;padding:8px;background:#141c2f;"
                f"border-radius:8px;border:1px solid {color}22'>"
                f"<b style='color:{color};font-size:1.3rem'>{val}</b>"
                f"<br><span style='font-size:.7rem;color:#64748b'>{lbl}</span></div>",
                unsafe_allow_html=True,
            )
    with col2:
        st.plotly_chart(fig_roc_pr(y_true, y_prob), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='font-size:1.1rem'>Classification Report</div>", unsafe_allow_html=True)
    report_dict = classification_report(y_true, m["y_pred"], target_names=["No Churn", "Churn"], output_dict=True)
    df_report = pd.DataFrame(report_dict).T.round(3)
    st.dataframe(
        df_report.style
            .background_gradient(cmap="Blues", vmin=0, vmax=1, subset=["precision", "recall", "f1-score"])
            .format("{:.3f}", subset=["precision", "recall", "f1-score"]),
        use_container_width=True,
    )

def page_feature_importance(model, feature_names: list[str]):
    st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)

    df_imp = get_feature_importances(model, feature_names)
    if df_imp.empty:
        st.warning("Feature importance tidak tersedia.")
        return

    available_types = df_imp["type"].unique().tolist()
    col1, col2, col3 = st.columns([1, 1, 1])
    imp_type  = col1.selectbox("Importance Type", available_types, index=0)
    top_n     = col2.slider("Top N Features", 5, len(feature_names), 15)
    ascending = col3.radio("Sort Direction", ["Top-Down", "Bottom-Up"], horizontal=True) == "Bottom-Up"

    st.plotly_chart(
        fig_feature_importance(df_imp, imp_type, top_n, ascending),
        use_container_width=True,
    )

    # Table
    st.markdown("#### Raw Scores")
    df_display = (
        df_imp[df_imp["type"] == imp_type]
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    df_display.index += 1
    st.dataframe(
        df_display.style.bar(subset=["importance"], color="#1d4ed8"),
        use_container_width=True, height=320,
    )


# ── Page: Threshold Tuning ────────────────────────────────────────────────────
def page_threshold_tuning(model, feature_names: list[str], global_threshold: float):
    st.markdown("<div class='section-header'>Threshold Tuning Simulator</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='info-box'>Geser slider untuk melihat dampak threshold terhadap semua metrik secara real-time.<br>"
        "<b>Tip bisnis:</b> Threshold rendah → lebih agresif menangkap churn (Recall ↑). "
        "Threshold tinggi → lebih presisi (Precision ↑). Pilih berdasarkan biaya bisnis False Negative vs False Positive.</div>",
        unsafe_allow_html=True,
    )

    threshold = st.slider("Active Threshold", 0.10, 0.90, float(global_threshold), 0.01, key="tuner_slider")

    # Generate sample data
    df_synth = load_sample_data(500, feature_names)
    rng = np.random.default_rng(99)
    y_true = rng.choice([0, 1], size=500, p=[0.73, 0.27])
    X_aligned = align_columns(df_synth.copy(), feature_names)

    try:
        y_prob = model.predict_proba(X_aligned)[:, 1]
    except Exception as e:
        st.error(f"Inferensi gagal: {e}")
        return

    m = compute_metrics(y_true, y_prob, threshold)

    # Real-time KPI row
    cols = st.columns(4)
    for col, (label, val) in zip(cols, [
        ("Precision", f"{m['precision']:.3f}"),
        ("Recall",    f"{m['recall']:.3f}"),
        ("F1-Score",  f"{m['f1']:.3f}"),
        ("Accuracy",  f"{m['accuracy']:.3f}"),
    ]):
        col.markdown(kpi_card(label, val), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig_threshold_curve(y_true, y_prob, threshold), use_container_width=True)
    with col_b:
        st.plotly_chart(fig_prediction_dist(m["y_pred"]), use_container_width=True)

    col_c, _ = st.columns([1, 1])
    with col_c:
        st.plotly_chart(fig_confusion_matrix(m["cm"]), use_container_width=True)


# ── Page: Single Prediction ───────────────────────────────────────────────────
def page_single_prediction(model, feature_names: list[str], threshold: float):
    st.markdown("<div class='section-header'>Single Prediction</div>", unsafe_allow_html=True)

    # ── Reset session state ──
    if "sp_reset" not in st.session_state:
        st.session_state["sp_reset"] = False

    if st.button("🔄 Reset Form"):
        for k in list(st.session_state.keys()):
            if k.startswith("sp_"):
                del st.session_state[k]
        st.session_state["sp_reset"] = True
        st.rerun()

    # ── Form ──
    st.markdown("#### Input Customer Data")
    with st.form("single_pred_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Demographics**")
            gender      = st.selectbox("Gender", ["Male", "Female"], key="sp_gender")
            senior      = st.selectbox("Senior Citizen", [0, 1], key="sp_senior")
            partner     = st.selectbox("Partner", ["Yes", "No"], key="sp_partner")
            dependents  = st.selectbox("Dependents", ["Yes", "No"], key="sp_dependents")
            tenure      = st.number_input("Tenure (months)", 0, 72, 12, key="sp_tenure")

        with col2:
            st.markdown("**📡 Services**")
            phone_line   = st.selectbox("Phone Line", ["Single Line", "Multiple Lines", "No Phone Service"], key="sp_phoneline")
            internet     = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="sp_internet")
            online_sec   = st.selectbox("Online Security", ["Yes", "No"], key="sp_sec")
            online_bak   = st.selectbox("Online Backup", ["Yes", "No"], key="sp_bak")
            dev_prot     = st.selectbox("Device Protection", ["Yes", "No"], key="sp_devprot")
            tech_sup     = st.selectbox("Tech Support", ["Yes", "No"], key="sp_techsup")
            stream_tv    = st.selectbox("Streaming TV", ["Yes", "No"], key="sp_tv")
            stream_mov   = st.selectbox("Streaming Movies", ["Yes", "No"], key="sp_mov")

        with col3:
            st.markdown("**💳 Billing**")
            contract     = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="sp_contract")
            paperless    = st.selectbox("Paperless Billing", ["Yes", "No"], key="sp_paper")
            payment      = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ], key="sp_payment")
            monthly_chg  = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0, 0.5, key="sp_monthly")
            total_chg    = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_chg * tenure), key="sp_total")
            has_internet = "Has Internet" if internet != "No" else "No Internet"

        submitted = st.form_submit_button("⚡ Predict Churn")

    if submitted:
        # Build row
        row: dict[str, int | float] = {f: 0 for f in feature_names}
        # Binary label-encoded (0/1)
        row["gender"]          = 1 if gender == "Male" else 0
        row["SeniorCitizen"]   = int(senior)
        row["Partner"]         = 1 if partner == "Yes" else 0
        row["Dependents"]      = 1 if dependents == "Yes" else 0
        row["OnlineSecurity"]  = 1 if online_sec == "Yes" else 0
        row["OnlineBackup"]    = 1 if online_bak == "Yes" else 0
        row["DeviceProtection"]= 1 if dev_prot == "Yes" else 0
        row["TechSupport"]     = 1 if tech_sup == "Yes" else 0
        row["StreamingTV"]     = 1 if stream_tv == "Yes" else 0
        row["StreamingMovies"] = 1 if stream_mov == "Yes" else 0
        row["PaperlessBilling"]= 1 if paperless == "Yes" else 0
        row["HasInternet"]     = 1 if has_internet == "Has Internet" else 0
        # Numeric
        row["tenure"]          = float(tenure)
        row["MonthlyCharges"]  = float(monthly_chg)
        row["TotalCharges"]    = float(total_chg)
        # OHE
        row[f"InternetService_{internet}"] = 1
        row[f"Contract_{contract}"]        = 1
        row[f"PaymentMethod_{payment}"]    = 1
        row[f"PhoneLineStatus_{phone_line}"] = 1

        df_input = pd.DataFrame([row])
        df_input = align_columns(df_input, feature_names)

        try:
            prob = float(model.predict_proba(df_input)[0, 1])
        except Exception as e:
            st.error(f"Prediksi gagal: {e}")
            return

        pred = int(prob >= threshold)
        st.session_state["sp_result"] = {"prob": prob, "pred": pred, "threshold": threshold}

    # ── Result ──
    if "sp_result" in st.session_state:
        res = st.session_state["sp_result"]
        prob, pred = res["prob"], res["pred"]
        st.markdown("---")
        st.markdown("#### 🎯 Prediction Result")

        col_r1, col_r2, col_r3 = st.columns(3)

        pct = f"{prob*100:.1f}%"
        css_cls = "prob-high" if prob >= 0.6 else ("prob-med" if prob >= threshold else "prob-low")
        col_r1.markdown(f"<div class='{css_cls}'>{pct}</div>", unsafe_allow_html=True)
        col_r1.markdown("<p style='text-align:center;color:#64748b;font-size:.8rem;margin-top:6px'>Churn Probability</p>", unsafe_allow_html=True)

        verdict_color = "#ef4444" if pred == 1 else "#22c55e"
        verdict_text  = "⚠️ CHURN" if pred == 1 else "✅ RETAIN"
        col_r2.markdown(
            f"<div style='text-align:center;padding:10px;background:rgba(255,255,255,.03);"
            f"border-radius:10px;border:1px solid {verdict_color}44'>"
            f"<b style='font-size:1.6rem;color:{verdict_color}'>{verdict_text}</b>"
            f"<br><span style='font-size:.75rem;color:#64748b'>Threshold: {threshold:.2f}</span></div>",
            unsafe_allow_html=True,
        )

        conf_pct = abs(prob - threshold) / max(threshold, 1 - threshold) * 100
        conf_label = "High" if conf_pct > 40 else ("Medium" if conf_pct > 15 else "Low")
        conf_color = "#22c55e" if conf_pct > 40 else ("#f59e0b" if conf_pct > 15 else "#ef4444")
        col_r3.markdown(
            f"<div style='text-align:center;padding:10px;background:rgba(255,255,255,.03);"
            f"border-radius:10px;border:1px solid {conf_color}44'>"
            f"<b style='font-size:1.6rem;color:{conf_color}'>{conf_label}</b>"
            f"<br><span style='font-size:.75rem;color:#64748b'>Confidence Level</span></div>",
            unsafe_allow_html=True,
        )

        # Probability gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 32, "color": "#60a5fa", "family": "JetBrains Mono"}},
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#334155"),
                bar=dict(color="#3b82f6"),
                bgcolor="#141c2f",
                steps=[
                    dict(range=[0, threshold*100], color="#1a2540"),
                    dict(range=[threshold*100, 100], color="#1e1025"),
                ],
                threshold=dict(line=dict(color="#ef4444", width=3), thickness=0.85, value=threshold*100),
            ),
        ))
        gauge.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(gauge, use_container_width=True)


# ── Page: Batch Prediction ────────────────────────────────────────────────────
def page_batch_prediction(model, feature_names: list[str], threshold: float):
    st.markdown("<div class='section-header'>Batch Prediction</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='info-box'>Upload CSV dengan kolom sesuai fitur model. "
        "Kolom <code>Churn</code> (opsional) akan digunakan sebagai ground truth.</div>",
        unsafe_allow_html=True,
    )

    # Template download
    template_df = pd.DataFrame(columns=feature_names)
    template_csv = template_df.to_csv(index=False)
    st.download_button(
        "📥 Download Template CSV",
        data=template_csv,
        file_name="churn_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload prediction CSV", type="csv", key="batch_upload")

    if not uploaded:
        st.info("Belum ada file diupload. Contoh menggunakan 50 baris sintetis:")
        df_preview = load_sample_data(50, feature_names).head(5)
        st.dataframe(df_preview, use_container_width=True)
        return

    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error membaca CSV: {e}")
        return

    y_true_col = None
    if "Churn" in df_raw.columns:
        y_true_col = df_raw["Churn"].values.astype(int)
        df_raw = df_raw.drop(columns=["Churn"])

    # Validate columns
    missing_cols = [c for c in feature_names if c not in df_raw.columns]
    extra_cols   = [c for c in df_raw.columns if c not in feature_names]
    if missing_cols:
        st.warning(f"⚠️ Kolom hilang (akan diisi 0): {missing_cols}")
    if extra_cols:
        st.info(f"ℹ️ Kolom extra diabaikan: {extra_cols}")

    with st.spinner("⚡ Running inference..."):
        time.sleep(0.3)
        try:
            X_aligned = align_columns(df_raw.copy(), feature_names)
            y_prob    = model.predict_proba(X_aligned)[:, 1]
            y_pred    = (y_prob >= threshold).astype(int)
        except Exception as e:
            st.error(f"Inferensi gagal: {e}")
            return

    st.success(f"✅ Selesai: {len(df_raw)} baris diproses.")

    result_df = df_raw.copy()
    result_df["Churn_Probability"] = y_prob.round(4)
    result_df["Churn_Prediction"]  = y_pred
    result_df["Prediction_Label"]  = result_df["Churn_Prediction"].map({0: "No Churn", 1: "Churn"})

    if y_true_col is not None:
        result_df["Churn_Actual"] = y_true_col

    # Summary metrics
    n_churn = int(y_pred.sum())
    n_total = len(y_pred)
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_card("Total Records", str(n_total)), unsafe_allow_html=True)
    c2.markdown(kpi_card("Predicted Churn", str(n_churn), f"{n_churn/n_total*100:.1f}%"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Avg Churn Prob", f"{y_prob.mean():.3f}"), unsafe_allow_True=True) if False else \
    c3.markdown(kpi_card("Avg Churn Prob", f"{y_prob.mean():.3f}"), unsafe_allow_html=True)

    # Distribution chart
    fig_hist = go.Figure(go.Histogram(
        x=y_prob,
        nbinsx=30,
        marker=dict(color="#3b82f6", line=dict(color="#1e2940", width=1)),
    ))
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#ef4444",
                       annotation_text=f"Threshold={threshold:.2f}",
                       annotation_font=dict(color="#ef4444"))
    fig_hist.update_layout(
        **PLOTLY_LAYOUT, height=280,
        xaxis_title="Churn Probability",
        yaxis_title="Count",
        title=dict(text="Probability Distribution", x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Result table
    st.markdown("#### Prediction Results")
    display_cols = ["Churn_Probability", "Churn_Prediction", "Prediction_Label"]
    if y_true_col is not None:
        display_cols.append("Churn_Actual")
    st.dataframe(
        result_df[display_cols].style
            .background_gradient(subset=["Churn_Probability"], cmap="RdYlGn_r", vmin=0, vmax=1),
        use_container_width=True, height=350,
    )

    # Download
    csv_out = result_df.to_csv(index=False)
    st.download_button(
        "📤 Download Result CSV",
        data=csv_out,
        file_name="churn_predictions.csv",
        mime="text/csv",
    )


# ── Page: Model Info ──────────────────────────────────────────────────────────
def page_model_info(model, threshold: float, feature_names: list[str]):
    st.markdown("<div class='section-header'>Model Info</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🤖 Model Architecture")
        params = {}
        try:
            params = model.get_params()
        except Exception:
            pass

        info = {
            "Model Type": "XGBoostClassifier",
            "Search Strategy": "RandomizedSearchCV (40 iter)",
            "CV Strategy": "StratifiedKFold (5-Fold)",
            "Scoring Metric": "F1-Score",
            "Optimal Threshold": f"{threshold:.2f}",
            "Feature Count": str(len(feature_names)),
            "Target Variable": "Churn (0=No, 1=Yes)",
        }
        for k, v in info.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:8px 12px;"
                f"border-bottom:1px solid #1e2940'>"
                f"<span style='color:#64748b'>{k}</span>"
                f"<span style='color:#e2e8f0;font-family:JetBrains Mono,monospace;font-size:.87rem'>{v}</span></div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### ⚙️ Best Hyperparameters")
        tunable = ["n_estimators", "max_depth", "learning_rate",
                   "subsample", "colsample_bytree", "gamma", "min_child_weight"]
        for p in tunable:
            val = params.get(p, "N/A")
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:8px 12px;"
                f"border-bottom:1px solid #1e2940'>"
                f"<span style='color:#64748b'>{p}</span>"
                f"<span style='color:#60a5fa;font-family:JetBrains Mono,monospace;font-size:.87rem'>{val}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📚 Preprocessing Pipeline (Notebook Summary)")
    steps = [
        ("1. Data Collection", "Telco Customer Churn — Kaggle (blastchar/telco-customer-churn)"),
        ("2. Feature Engineering", "PhoneLineStatus (3-way), HasInternet flag, replace 'No internet service'"),
        ("3. Encoding", "LabelEncoder untuk binary cols, get_dummies untuk multi-valued cols"),
        ("4. Imbalance Handling", "SMOTE oversampling (random_state=42)"),
        ("5. Threshold Tuning", "Sweep 0.30–0.70 step 0.01, maximize F1-Score"),
        ("6. Model Saving", "joblib.dump(model, 'model/xgb_churn_model.pkl')"),
    ]
    for step, desc in steps:
        st.markdown(
            f"<div style='padding:10px 14px;background:#141c2f;border-radius:8px;"
            f"margin-bottom:8px;border-left:3px solid #3b82f6'>"
            f"<b style='color:#93c5fd'>{step}</b>"
            f"<br><span style='font-size:.85rem;color:#94a3b8'>{desc}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💡 Business Recommendations")
    recs = [
        ("🎯 Threshold < 0.38", "Lebih agresif. Cocok jika biaya False Negative (kehilangan pelanggan) > biaya retensi."),
        ("🛡️ Threshold > 0.38", "Lebih selektif. Gunakan jika tim retensi terbatas dan ingin precision tinggi."),
        ("📊 Monitor F1 & Recall", "Di industri telco, Recall lebih kritis — jangan biarkan churn lolos."),
        ("🔄 Retrain berkala", "Distribusi pelanggan berubah. Retrain minimal setiap kuartal."),
    ]
    for title, desc in recs:
        st.markdown(
            f"<div style='padding:10px 14px;background:#141c2f;border-radius:8px;"
            f"margin-bottom:8px;border-left:3px solid #f59e0b'>"
            f"<b style='color:#fcd34d'>{title}</b>"
            f"<br><span style='font-size:.85rem;color:#94a3b8'>{desc}</span></div>",
            unsafe_allow_html=True,
        )


# ── Main App Entry ─────────────────────────────────────────────────────────────
def main():
    # Load model
    model, threshold_pkl, load_error = load_model_and_threshold()

    # Session state init
    if "active_threshold" not in st.session_state:
        st.session_state["active_threshold"] = threshold_pkl

    # Sidebar
    page, global_thresh = render_sidebar(st.session_state["active_threshold"])
    st.session_state["active_threshold"] = global_thresh
    threshold = global_thresh

    # Error banner
    if load_error:
        st.error(f"⚠️ {load_error}")
        st.info(
            "Pastikan file model ada di `model/xgb_churn_model.pkl`.\n\n"
            "Struktur folder:\n```\nproject/\n├── app.py\n└── model/\n    ├── xgb_churn_model.pkl\n    └── xgb_churn_threshold.pkl\n```"
        )
        return

    feature_names = get_feature_names(model)

    # Route
    if page == "📊  Dashboard Overview":
        page_dashboard(model, feature_names, threshold)
    elif page == "📈  Feature Importance":
        page_feature_importance(model, feature_names)
    elif page == "🎯  Threshold Tuning":
        page_threshold_tuning(model, feature_names, threshold)
    elif page == "🔮  Single Prediction":
        page_single_prediction(model, feature_names, threshold)
    elif page == "📦  Batch Prediction":
        page_batch_prediction(model, feature_names, threshold)
    elif page == "ℹ️   Model Info":
        page_model_info(model, threshold, feature_names)


if __name__ == "__main__":
    main()