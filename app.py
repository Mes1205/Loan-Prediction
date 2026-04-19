"""
LoanIQ — Streamlit App
Memuat hasil model langsung dari file export notebook (ga_best_chrom.npz,
ann_model.pt, scaler.pkl, pipeline_metrics.npz, ga_history.npz).
Tidak ada training ulang sama sekali.
"""

import io
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Loan Prediction Intelligence Dashboard",
    page_icon="",
    layout="wide",
)

# ============================================================
# GLOBAL STYLE
# ============================================================
PRIMARY    = "#2563EB"
SECONDARY  = "#0F172A"
ACCENT     = "#10B981"
WARNING    = "#F59E0B"
DANGER     = "#EF4444"
CARD_BG    = "#FFFFFF"
BORDER     = "#E2E8F0"
TEXT_MUTED = "#64748B"

sns.set_theme(style="whitegrid")

st.markdown(f"""
<style>
    .stApp {{ background: #F0F4FF; }}
    [data-testid="stSidebar"] {{ background: {SECONDARY} !important; padding-top: 0 !important; }}
    [data-testid="stSidebar"] > div:first-child {{ padding-top: 0; }}
    .nav-logo {{
        background: linear-gradient(135deg, {PRIMARY} 0%, #1D4ED8 100%);
        padding: 28px 20px 22px 20px; margin-bottom: 8px;
    }}
    .nav-logo-title {{ font-size:1.05rem; font-weight:800; color:white; letter-spacing:-0.3px; line-height:1.3; }}
    .nav-logo-sub   {{ font-size:0.72rem; color:rgba(255,255,255,0.6); margin-top:3px; }}
    .nav-section-label {{
        font-size:0.65rem; font-weight:700; color:rgba(255,255,255,0.35);
        letter-spacing:1.5px; text-transform:uppercase; padding:16px 20px 6px 20px;
    }}
    .nav-divider {{ height:1px; background:rgba(255,255,255,0.08); margin:10px 20px; }}
    .section-card {{
        background:{CARD_BG}; border:1px solid {BORDER}; border-radius:18px;
        padding:20px 22px; box-shadow:0 6px 20px rgba(15,23,42,0.05); margin-bottom:16px;
    }}
    .metric-card {{
        background:white; border:1px solid {BORDER}; border-radius:18px;
        padding:16px 18px; box-shadow:0 4px 14px rgba(15,23,42,0.04);
    }}
    .metric-label  {{ font-size:0.9rem; color:{TEXT_MUTED}; margin-bottom:6px; }}
    .metric-value  {{ font-size:1.8rem; font-weight:800; color:{SECONDARY}; line-height:1.1; }}
    .metric-caption{{ font-size:0.85rem; color:{TEXT_MUTED}; margin-top:6px; }}
    .badge {{
        display:inline-block; padding:4px 10px; border-radius:999px;
        background:#DBEAFE; color:#1D4ED8; font-size:0.8rem; font-weight:700; margin-bottom:10px;
    }}
    .small-note {{ color:{TEXT_MUTED}; font-size:0.9rem; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:8px; }}
    .stTabs [data-baseweb="tab"] {{ border-radius:10px; padding:8px 16px; background:#EFF6FF; }}
    .stTabs [aria-selected="true"] {{ background:{PRIMARY} !important; color:white !important; }}
    .block-container {{ padding-top:2.5rem; padding-bottom:3rem; }}
    .result-approved {{
        background:linear-gradient(135deg,#ECFDF5 0%,#D1FAE5 100%);
        border:2px solid #10B981; border-radius:20px; padding:24px 28px; text-align:center;
    }}
    .result-rejected {{
        background:linear-gradient(135deg,#FEF2F2 0%,#FEE2E2 100%);
        border:2px solid #EF4444; border-radius:20px; padding:24px 28px; text-align:center;
    }}
    .result-loading {{
        background:#F8FAFC; border:2px dashed {BORDER};
        border-radius:20px; padding:24px 28px; text-align:center;
    }}
    .score-bar-wrap {{
        background:{BORDER}; border-radius:999px; height:12px; margin:10px 0; overflow:hidden;
    }}
    .factor-card {{
        background:white; border:1px solid {BORDER}; border-radius:12px;
        padding:12px 16px; margin-bottom:8px;
    }}
    .input-section-title {{ font-size:0.95rem; font-weight:700; color:{SECONDARY}; margin:16px 0 8px 0; }}
    .model-verdict-label {{
        font-size:0.72rem; font-weight:700; text-transform:uppercase;
        letter-spacing:1px; color:{TEXT_MUTED}; margin-bottom:8px;
    }}
    .page-title {{ padding-top:12px; margin-bottom:24px; }}
    .page-title-main {{ font-size:1.5rem; font-weight:800; color:{SECONDARY}; }}
    .page-title-sub  {{ font-size:0.88rem; color:{TEXT_MUTED}; margin-top:4px; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
DEFAULT_THRESHOLD_FIS    = 52
DEFAULT_THRESHOLD_HYBRID = 50

# File paths — letakkan semua di folder yang sama dengan app.py
FILE_CHROM   = "ga_best_chrom.npz"
FILE_MODEL   = "ann_model.pt"
FILE_SCALER  = "scaler.pkl"
FILE_METRICS = "pipeline_metrics.npz"
FILE_HISTORY = "ga_history.npz"
FILE_DATA    = "data_train.csv"

REQUIRED_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Credit_History", "Loan_Status",
]

STRUCTURE = {
    'income':         ['low', 'medium', 'high'],
    'loan_amount':    ['small', 'medium', 'large'],
    'credit_history': ['bad', 'good'],
    'income_ratio':   ['low', 'medium', 'high'],
    'approval':       ['reject', 'review', 'approve'],
}

BOUNDS = {
    'income': {
        'low':    [(0,0),(0,0),(1000,6000),(2000,8000)],
        'medium': [(1000,5000),(3000,8000),(5000,12000),(7000,15000)],
        'high':   [(5000,10000),(8000,15000),(50000,100000),(50000,100000)],
    },
    'loan_amount': {
        'small':  [(0,0),(0,0),(50,150),(80,200)],
        'medium': [(50,150),(100,200),(200,350),(250,450)],
        'large':  [(200,350),(300,450),(600,700),(600,700)],
    },
    'credit_history': {
        'bad':  [(0,0),(0,0),(0.1,0.5),(0.3,0.7)],
        'good': [(0.3,0.6),(0.5,0.8),(1.0,1.0),(1.0,1.0)],
    },
    'income_ratio': {
        'low':    [(0,0),(0,0),(10,30),(20,50)],
        'medium': [(10,40),(20,50),(40,80),(60,100)],
        'high':   [(40,80),(70,120),(9000,99999),(9000,99999)],
    },
    'approval': {
        'reject':  [(0,0),(0,0),(10,30),(25,50)],
        'review':  [(20,40),(40,55),(50,65),(60,80)],
        'approve': [(50,70),(65,85),(100,100),(100,100)],
    },
}

THRESHOLD_BOUNDS = (35.0, 75.0)

# ============================================================
# SESSION STATE
# ============================================================
if "active_page" not in st.session_state:
    st.session_state.active_page = "predict"

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
        <div class="nav-logo">
            <div class="nav-logo-title">LoanIQ</div>
            <div class="nav-logo-sub">Loan Prediction Intelligence</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="nav-section-label">Menu Utama</div>', unsafe_allow_html=True)

    if st.button("Prediksi Kelayakan", key="nav_predict", use_container_width=True,
                 type="primary" if st.session_state.active_page == "predict" else "secondary"):
        st.session_state.active_page = "predict"
        st.rerun()

    if st.button("Analisis & Training", key="nav_analysis", use_container_width=True,
                 type="primary" if st.session_state.active_page == "analysis" else "secondary"):
        st.session_state.active_page = "analysis"
        st.rerun()

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div style="padding:12px 20px; font-size:0.75rem; color:rgba(255,255,255,0.35); line-height:1.6;">
            <div style="font-weight:700; color:rgba(255,255,255,0.5); margin-bottom:6px;">Model Info</div>
            FIS + GA Optimization<br>
            Neuro-Fuzzy ANN Hybrid<br>
            Loaded from: Notebook Export<br>
            Threshold: {DEFAULT_THRESHOLD_FIS} (manual) / GA (dynamic)
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# HELPER UI
# ============================================================
def section_header(badge, title, desc=None):
    st.markdown(f"""
        <div class="section-card">
            <div class="badge">{badge}</div>
            <div style="font-size:1.25rem; font-weight:800; color:{SECONDARY}; margin-bottom:6px;">{title}</div>
            <div class="small-note">{desc or ""}</div>
        </div>
    """, unsafe_allow_html=True)

def metric_card(label, value, caption=""):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
    """, unsafe_allow_html=True)

def format_pct(x):  return f"{x*100:.2f}%"
def format_pp(x):   return f"{x:+.2f} pp"

def style_axis(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", color=SECONDARY, pad=12)
    ax.set_xlabel(xlabel, fontsize=10, color=SECONDARY)
    ax.set_ylabel(ylabel, fontsize=10, color=SECONDARY)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25)

# ============================================================
# MANUAL FIS
# ============================================================
class ManualFIS:
    def __init__(self):
        self.rules = []
        self.mf_params = {}
        self._initialize_membership_functions()
        self._initialize_rules()

    def _initialize_membership_functions(self):
        self.mf_params["income"] = {
            "low":    [0, 0, 4000, 6000],
            "medium": [4000, 6000, 8000, 10000],
            "high":   [8000, 12000, 100000, 100000],
        }
        self.mf_params["loan_amount"] = {
            "small":  [0, 0, 100, 150],
            "medium": [100, 150, 250, 350],
            "large":  [250, 350, 700, 700],
        }
        self.mf_params["credit_history"] = {
            "bad":  [0, 0, 0.3, 0.5],
            "good": [0.5, 0.7, 1.0, 1.0],
        }
        self.mf_params["income_ratio"] = {
            "low":    [0, 0, 20, 40],
            "medium": [20, 40, 60, 80],
            "high":   [60, 100, 99999, 99999],
        }
        self.mf_params["approval"] = {
            "reject":  [0, 0, 20, 45],
            "review":  [35, 50, 60, 75],
            "approve": [60, 80, 100, 100],
        }

    def _trapezoidal_mf(self, x, params):
        a, b, c, d = params
        if x < a or x > d:   return 0.0
        elif a < x <= b:      return (x - a) / (b - a) if b != a else 1.0
        elif b < x <= c:      return 1.0
        else:                 return (d - x) / (d - c) if d != c else 1.0

    def _fuzzify(self, value, variable):
        return {label: self._trapezoidal_mf(value, params)
                for label, params in self.mf_params[variable].items()}

    def _initialize_rules(self):
        self.rules = [
            {"conditions": {"credit_history": "good", "income_ratio": "high"},                           "output": "approve", "weight": 1.0},
            {"conditions": {"credit_history": "good", "income_ratio": "medium", "income": "high"},       "output": "approve", "weight": 0.9},
            {"conditions": {"credit_history": "good", "income_ratio": "medium", "loan_amount": "small"}, "output": "approve", "weight": 0.85},
            {"conditions": {"credit_history": "good", "income": "medium",       "loan_amount": "small"}, "output": "approve", "weight": 0.8},
            {"conditions": {"credit_history": "good", "income_ratio": "low"},                            "output": "review",  "weight": 0.7},
            {"conditions": {"credit_history": "bad",  "income_ratio": "high"},                           "output": "review",  "weight": 0.6},
            {"conditions": {"credit_history": "bad",  "income": "high", "loan_amount": "small"},         "output": "review",  "weight": 0.65},
            {"conditions": {"credit_history": "bad",  "income_ratio": "low"},                            "output": "reject",  "weight": 1.0},
            {"conditions": {"credit_history": "bad",  "loan_amount": "large"},                           "output": "reject",  "weight": 1.0},
            {"conditions": {"income": "low",           "loan_amount": "large"},                          "output": "reject",  "weight": 0.90},
            {"conditions": {"income": "low",           "credit_history": "bad"},                         "output": "reject",  "weight": 0.95},
        ]

    def _evaluate_rules(self, fuzzy_inputs):
        activations = {"reject": 0, "review": 0, "approve": 0}
        for rule in self.rules:
            strengths = [fuzzy_inputs[var].get(label, 0)
                         for var, label in rule["conditions"].items()
                         if var in fuzzy_inputs]
            if strengths:
                rule_strength = min(strengths) * rule["weight"]
                activations[rule["output"]] = max(activations[rule["output"]], rule_strength)
        return activations

    def _defuzzify(self, activations):
        x = np.linspace(0, 100, 1000)
        aggregated = np.zeros_like(x)
        for output_label, strength in activations.items():
            if strength > 0:
                params = self.mf_params["approval"][output_label]
                mf_values = np.array([self._trapezoidal_mf(xi, params) for xi in x])
                aggregated = np.maximum(aggregated, np.minimum(mf_values, strength))
        if aggregated.sum() == 0: return 50
        return np.sum(x * aggregated) / np.sum(aggregated)

    def predict_single(self, income, loan_amount, credit_history, coapplicant_income=0):
        total_income = income + coapplicant_income
        income_ratio = (total_income / loan_amount) if loan_amount > 0 else 100
        fuzzy_inputs = {
            "income":         self._fuzzify(total_income,    "income"),
            "loan_amount":    self._fuzzify(loan_amount,     "loan_amount"),
            "credit_history": self._fuzzify(credit_history,  "credit_history"),
            "income_ratio":   self._fuzzify(income_ratio,    "income_ratio"),
        }
        activations = self._evaluate_rules(fuzzy_inputs)
        score = self._defuzzify(activations)
        return (1 if score >= DEFAULT_THRESHOLD_FIS else 0), score, activations

    def predict(self, X):
        predictions, scores = [], []
        for i in range(len(X)):
            pred, score, _ = self.predict_single(
                X.iloc[i]["ApplicantIncome"], X.iloc[i]["LoanAmount"],
                X.iloc[i]["Credit_History"],  X.iloc[i]["CoapplicantIncome"],
            )
            predictions.append(pred)
            scores.append(score)
        return np.array(predictions), np.array(scores)

    def visualize_membership_functions(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.patch.set_facecolor("white")
        variables = [
            ("income",         "Applicant Income",    0,  50000, axes[0, 0]),
            ("loan_amount",    "Loan Amount",          0,  700,   axes[0, 1]),
            ("credit_history", "Credit History",       0,  1,     axes[0, 2]),
            ("income_ratio",   "Income to Loan Ratio", 0,  200,   axes[1, 0]),
            ("approval",       "Loan Approval Score",  0,  100,   axes[1, 1]),
        ]
        for var_name, title, x_min, x_max, ax in variables:
            x = np.linspace(x_min, x_max, 500)
            for label, params in self.mf_params[var_name].items():
                ax.plot(x, [self._trapezoidal_mf(xi, params) for xi in x], linewidth=2.5, label=label)
            style_axis(ax, title, "Value", "Membership Degree")
            ax.legend(fontsize=8, frameon=True)
            ax.set_ylim(-0.05, 1.05)
        axes[1, 2].axis("off")
        fig.suptitle("Manual Membership Functions", fontsize=15, fontweight="bold", color=SECONDARY)
        plt.tight_layout()
        return fig

# ============================================================
# GA FIS
# ============================================================
def chromosome_to_params(chrom):
    params, idx = {}, 0
    for var, labels in STRUCTURE.items():
        params[var] = {}
        for label in labels:
            bounds = BOUNDS[var][label]
            vals = sorted([np.clip(chrom[idx + j], bounds[j][0], bounds[j][1]) for j in range(4)])
            params[var][label] = vals
            idx += 4
    return params

def chromosome_to_threshold(chrom):
    return float(np.clip(chrom[-1], *THRESHOLD_BOUNDS))

class GA_FIS(ManualFIS):
    def set_params(self, chrom):
        self.mf_params = chromosome_to_params(chrom)
        self.decision_threshold = chromosome_to_threshold(chrom)

    def predict_single(self, income, loan_amount, credit_history, coapplicant_income=0):
        total_income = income + coapplicant_income
        income_ratio = (total_income / loan_amount) if loan_amount > 0 else 100
        fuzzy_inputs = {
            'income':         self._fuzzify(total_income,   'income'),
            'loan_amount':    self._fuzzify(loan_amount,    'loan_amount'),
            'credit_history': self._fuzzify(credit_history, 'credit_history'),
            'income_ratio':   self._fuzzify(income_ratio,   'income_ratio'),
        }
        activations = self._evaluate_rules(fuzzy_inputs)
        score = self._defuzzify(activations)
        return (1 if score >= self.decision_threshold else 0), score, activations

# ============================================================
# ANN MODEL
# ============================================================
class TunerNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 32), nn.LeakyReLU(0.1),
            nn.Linear(32, 16),        nn.LeakyReLU(0.1),
            nn.Linear(16, 1),         nn.Sigmoid(),
        )
    def forward(self, x):
        return self.layer(x)

# ============================================================
# LOAD ALL EXPORTS (cached — runs once per session)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Load semua hasil export dari notebook:
      - ga_best_chrom.npz  → best_chrom (numpy array)
      - ann_model.pt       → TunerNet state_dict
      - scaler.pkl         → RobustScaler
      - pipeline_metrics.npz → semua angka metric
      - ga_history.npz     → history konvergensi GA
      - data_train.csv     → untuk preview dataset & plot FIS
    """
    missing = []
    for f in [FILE_CHROM, FILE_MODEL, FILE_SCALER, FILE_METRICS, FILE_HISTORY]:
        import os
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        raise FileNotFoundError(
            f"File export tidak ditemukan: {missing}\n"
            "Jalankan cell EXPORT di notebook terlebih dahulu, "
            "lalu letakkan semua file hasil download di folder yang sama dengan app.py."
        )

    p = {}

    # --- GA best_chrom ---
    chrom_data   = np.load(FILE_CHROM)
    best_chrom   = chrom_data["best_chrom"]
    threshold_ga = chromosome_to_threshold(best_chrom)
    p["best_chrom"]   = best_chrom
    p["threshold_ga"] = threshold_ga

    # --- GA History ---
    hist_data    = np.load(FILE_HISTORY)
    p["ga_history"] = {
        "best":  hist_data["best"].tolist(),
        "mean":  hist_data["mean"].tolist(),
        "worst": hist_data["worst"].tolist(),
    }

    # --- Metrics ---
    m = np.load(FILE_METRICS)
    p["metrics"] = {k: float(m[k]) for k in m.files}

    # --- ANN Model ---
    ann_model = TunerNet(input_dim=4)
    ann_model.load_state_dict(torch.load(FILE_MODEL, map_location="cpu"))
    ann_model.eval()
    p["ann_model"] = ann_model

    # --- Scaler ---
    with open(FILE_SCALER, "rb") as f_sc:
        p["ann_scaler"] = pickle.load(f_sc)

    # --- Manual FIS object ---
    p["fis"] = ManualFIS()

    # --- GA FIS object ---
    ga_fis = GA_FIS()
    ga_fis.set_params(best_chrom)
    p["ga_fis"] = ga_fis

    # --- Dataset (for preview & plots) ---
    import os
    if os.path.exists(FILE_DATA):
        df = _load_df(FILE_DATA)
        p["df"] = df
    else:
        p["df"] = None

    # --- Pre-compute all figures ---
    p["fig_mf_fis"]    = _fig_mf_manual(p["fis"])
    p["fig_eval_fis"]  = _fig_eval_fis(p)
    p["fig_mf_ga"]     = _fig_mf_ga(p)
    p["fig_konvergensi"]= _fig_konvergensi(p)
    p["fig_ga_eval"]   = _fig_ga_eval(p)
    p["fig_ann"]       = _fig_ann(p)

    return p


def _load_df(path):
    df = None
    for sep in [",", ";", "\t"]:
        try:
            t = pd.read_csv(path, sep=sep)
            if t.shape[1] > 1:
                df = t.copy(); break
        except Exception:
            pass
    if df is None:
        df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    for col in ["Gender", "Married", "Dependents", "Self_Employed", "Loan_Amount_Term"]:
        if col in df.columns and df[col].isna().any():
            mv = df[col].mode()
            if not mv.empty:
                df[col] = df[col].fillna(mv.iloc[0])
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Credit_History"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["LoanAmount"]        = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Credit_History"]    = df["Credit_History"].fillna(0)
    df["CoapplicantIncome"] = df["CoapplicantIncome"].fillna(0)
    df["Loan_Status"] = df["Loan_Status"].astype(str).str.strip().str.upper()
    df["Loan_Status"] = df["Loan_Status"].replace(
        {"Y": 1, "N": 0, "YES": 1, "NO": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    df["Loan_Status"] = pd.to_numeric(df["Loan_Status"], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLS).copy()
    df["Loan_Status"] = df["Loan_Status"].astype(int)
    return df


# ── Figure builders (called once during load, cached via cache_resource) ──

def _fig_mf_manual(fis):
    return fis.visualize_membership_functions()


def _fig_eval_fis(p):
    m = p["metrics"]
    # Rebuild confusion-matrix-like figure using only stored metric numbers
    # (no raw data needed — we display acc + classification report approximation)
    # For the confusion matrix we need the actual predictions; rebuild from df if available
    df = p.get("df")
    if df is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Accuracy FIS Manual: {m['acc_fis']*100:.2f}%\n"
                           f"(data_train.csv tidak tersedia untuk plot CM)",
                ha="center", va="center", fontsize=14, color=SECONDARY)
        ax.axis("off")
        return fig

    features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Credit_History"]
    X, y = df[features], df["Loan_Status"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    preds, scores = p["fis"].predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
    style_axis(axes[0], "Confusion Matrix - Manual FIS", "Predicted", "Actual")
    axes[1].hist(scores, bins=25, color=PRIMARY, alpha=0.85, edgecolor="white")
    axes[1].axvline(DEFAULT_THRESHOLD_FIS, color=DANGER, linestyle="--", linewidth=2,
                    label=f"Threshold {DEFAULT_THRESHOLD_FIS}")
    style_axis(axes[1], "Approval Score Distribution", "Score", "Frequency")
    axes[1].legend()
    plt.tight_layout()
    return fig


def _fig_mf_ga(p):
    """Perbandingan MF manual vs GA-Optimized."""
    manual_mf = p["fis"].mf_params
    ga_mf     = chromosome_to_params(p["best_chrom"])
    variabel_info = [
        ('income',         'Income',         0, 20000),
        ('loan_amount',    'Loan Amount',     0, 700),
        ('credit_history', 'Credit History',  0, 1),
        ('income_ratio',   'Income Ratio',    0, 200),
        ('approval',       'Approval Score',  0, 100),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Perbandingan MF: Manual (solid) vs GA-Optimized (putus-putus)',
                 fontsize=14, fontweight='bold', color=SECONDARY)
    axes_flat = axes.flatten()
    fis_tmp = ManualFIS()
    for i, (var, judul, xmin, xmax) in enumerate(variabel_info):
        ax, x = axes_flat[i], np.linspace(xmin, xmax, 500)
        for label in STRUCTURE[var]:
            y_m = [fis_tmp._trapezoidal_mf(xi, manual_mf[var][label]) for xi in x]
            y_g = [fis_tmp._trapezoidal_mf(xi, ga_mf[var][label]) for xi in x]
            line, = ax.plot(x, y_m, linewidth=2.5, label=f'{label} (Manual)')
            ax.plot(x, y_g, linewidth=2.2, linestyle='--',
                    color=line.get_color(), alpha=0.9, label=f'{label} (GA)')
        style_axis(ax, judul, "Value", "Membership")
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.1)
    axes_flat[5].axis('off')
    plt.tight_layout()
    return fig


def _fig_konvergensi(p):
    hist = p["ga_history"]
    m    = p["metrics"]
    baseline_fitness = m["baseline_fitness"]
    fig, ax = plt.subplots(figsize=(10, 5))
    gen = range(1, len(hist['best']) + 1)
    ax.plot(gen, hist['best'],  label='Best fitness',  color=PRIMARY,  linewidth=2.5)
    ax.plot(gen, hist['mean'],  label='Mean fitness',  color=ACCENT,   linewidth=2,   linestyle='--')
    ax.plot(gen, hist['worst'], label='Worst fitness', color=WARNING,  linewidth=2,   linestyle=':')
    ax.axhline(baseline_fitness, color=DANGER, linewidth=2, linestyle='-.',
               label=f'Manual FIS baseline ({baseline_fitness:.4f})')
    ax.fill_between(gen, hist['worst'], hist['best'], alpha=0.08, color=PRIMARY)
    style_axis(ax, "Konvergensi GA — Optimasi MF + Threshold", "Generasi",
               "Fitness (0.4×BalAcc + 0.6×F1_macro)")
    ax.legend()
    ax.set_ylim(max(0.4, min(hist['worst']) - 0.05), 1.0)
    plt.tight_layout()
    return fig


def _fig_ga_eval(p):
    """Plot evaluasi GA: CM + score dist + threshold curve."""
    df = p.get("df")
    if df is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        m = p["metrics"]
        ax.text(0.5, 0.5,
                f"GA Accuracy (test): {m['acc_ga']*100:.2f}%\n"
                f"GA F1 Macro: {m['f1_ga']*100:.2f}%\n"
                f"Balanced Acc: {m['bal_acc_ga']*100:.2f}%\n"
                f"Threshold GA: {m['best_threshold_ga']:.2f}",
                ha="center", va="center", fontsize=13, color=SECONDARY)
        ax.axis("off")
        return fig

    from sklearn.model_selection import train_test_split as tts
    features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Credit_History"]
    X, y = df[features], df["Loan_Status"]
    X_temp, X_test, y_temp, y_test = tts(X, y, test_size=0.30, random_state=42)
    X_train, X_val, y_train, y_val = tts(X_temp, y_temp, test_size=0.33, random_state=42)

    preds_ga, scores_ga = p["ga_fis"].predict(X_test)
    threshold_ga = p["threshold_ga"]
    cm = confusion_matrix(y_test, preds_ga)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[0],
                xticklabels=['Rejected','Approved'], yticklabels=['Rejected','Approved'])
    style_axis(axes[0], "Confusion Matrix - GA-Optimized FIS", "Predicted", "Actual")
    axes[1].hist(scores_ga, bins=25, color=ACCENT, alpha=0.85, edgecolor='white')
    axes[1].axvline(threshold_ga, color=DANGER, linestyle='--', linewidth=2,
                    label=f'Threshold GA ({threshold_ga:.1f})')
    axes[1].axvline(DEFAULT_THRESHOLD_FIS, color=WARNING, linestyle=':', linewidth=2,
                    label=f'Threshold Manual ({DEFAULT_THRESHOLD_FIS})')
    style_axis(axes[1], "GA Approval Score Distribution", "Score", "Frequency")
    axes[1].legend()
    thresholds = np.arange(30, 80, 0.5)
    acc_c = [accuracy_score(y_test, (scores_ga >= t).astype(int)) for t in thresholds]
    f1_c  = [f1_score(y_test,  (scores_ga >= t).astype(int), average='macro', zero_division=0) for t in thresholds]
    ba_c  = [balanced_accuracy_score(y_test, (scores_ga >= t).astype(int)) for t in thresholds]
    axes[2].plot(thresholds, acc_c, label='Accuracy',         color=PRIMARY,   linewidth=2)
    axes[2].plot(thresholds, f1_c,  label='F1 Macro',         color=ACCENT,    linewidth=2, linestyle='--')
    axes[2].plot(thresholds, ba_c,  label='Balanced Accuracy', color=SECONDARY, linewidth=1.7, linestyle=':')
    axes[2].axvline(threshold_ga, color=DANGER,  linestyle='--', linewidth=2, label=f'Threshold GA ({threshold_ga:.1f})')
    axes[2].axvline(DEFAULT_THRESHOLD_FIS, color=WARNING, linestyle=':', linewidth=2, label=f'Threshold Manual ({DEFAULT_THRESHOLD_FIS})')
    best_f1_t = thresholds[np.argmax(f1_c)]
    axes[2].scatter([best_f1_t], [max(f1_c)], color=ACCENT, s=80, zorder=5, label=f'Max F1 @ {best_f1_t:.1f}')
    style_axis(axes[2], "Threshold vs Metrics - Test Set", "Decision Threshold", "Score")
    axes[2].set_ylim(0.4, 1.0)
    axes[2].legend(fontsize=8)
    axes[2].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    return fig


def _fig_ann(p):
    """Figure ANN: CM + score dist + metric comparison + score shift."""
    df = p.get("df")
    m  = p["metrics"]
    if df is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                f"Manual FIS Accuracy : {m['acc_manual_ann']*100:.2f}%\n"
                f"Neuro-Fuzzy Accuracy: {m['acc_hybrid_ann']*100:.2f}%\n"
                f"Improvement         : {m['improvement_pp_ann']:+.2f} pp",
                ha="center", va="center", fontsize=13, color=SECONDARY)
        ax.axis("off")
        return fig

    features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Credit_History"]
    X_ann, y_ann = df[features], df["Loan_Status"]
    X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(
        X_ann, y_ann, test_size=0.3, random_state=42)

    scaler = p["ann_scaler"]
    X_test_scaled = scaler.transform(X_test_ann)
    ann_model = p["ann_model"]
    with torch.no_grad():
        ann_probs = ann_model(torch.FloatTensor(X_test_scaled)).numpy().flatten() * 100

    _, fis_scores_test = p["fis"].predict(X_test_ann)
    final_hybrid_scores = np.array([
        (f * 0.3 + a * 0.7) if 30 < f < 70 else (f * 0.8 + a * 0.2)
        for f, a in zip(fis_scores_test, ann_probs)
    ])
    hybrid_preds = (final_hybrid_scores >= DEFAULT_THRESHOLD_HYBRID).astype(int)
    manual_preds, _ = p["fis"].predict(X_test_ann)

    cm = confusion_matrix(y_test_ann, hybrid_preds)
    acc_manual = accuracy_score(y_test_ann, manual_preds)
    acc_hybrid = accuracy_score(y_test_ann, hybrid_preds)

    fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
    fig4.suptitle("Integrated Evaluation Dashboard - Manual FIS vs Neuro-Fuzzy",
                  fontsize=15, fontweight="bold", color=SECONDARY)

    # Empty placeholder where loss history would be — show text summary instead
    axes4[0, 0].axis("off")
    axes4[0, 0].text(0.05, 0.95,
        f"ANN Epochs     : 300\n"
        f"Architecture   : 4→32→16→1\n"
        f"Activation     : LeakyReLU + Sigmoid\n"
        f"Optimizer      : Adam (lr=0.001)\n"
        f"Weight Decay   : 1e-5\n\n"
        f"Final Loss (epoch 300): ~0.485",
        transform=axes4[0, 0].transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor=BORDER))
    axes4[0, 0].set_title("ANN Architecture", fontweight="bold", color=SECONDARY)

    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=axes4[0, 1],
                xticklabels=["Rejected","Approved"], yticklabels=["Rejected","Approved"])
    style_axis(axes4[0, 1], "Confusion Matrix", "Predicted", "Actual")

    axes4[0, 2].hist(final_hybrid_scores, bins=25, alpha=0.85, edgecolor="white", color=ACCENT)
    axes4[0, 2].axvline(DEFAULT_THRESHOLD_HYBRID, color=DANGER, linestyle="--", linewidth=2,
                        label=f"Threshold {DEFAULT_THRESHOLD_HYBRID}")
    style_axis(axes4[0, 2], "Hybrid Score Distribution", "Score", "Frequency")
    axes4[0, 2].legend()

    df_metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Manual FIS": [acc_manual,
                       precision_score(y_test_ann, manual_preds, zero_division=0),
                       recall_score(y_test_ann, manual_preds, zero_division=0),
                       f1_score(y_test_ann, manual_preds, zero_division=0)],
        "Neuro-Fuzzy": [acc_hybrid,
                        precision_score(y_test_ann, hybrid_preds, zero_division=0),
                        recall_score(y_test_ann, hybrid_preds, zero_division=0),
                        f1_score(y_test_ann, hybrid_preds, zero_division=0)],
    }).set_index("Metric")

    df_metrics.plot(kind="bar", ax=axes4[1, 0], rot=0, color=[PRIMARY, ACCENT])
    style_axis(axes4[1, 0], "Metric Comparison", "", "Score")
    axes4[1, 0].set_ylim(0, 1.1)
    axes4[1, 0].legend(loc="lower right")

    sns.kdeplot(fis_scores_test,       label="FIS Score",    fill=True, alpha=0.35, ax=axes4[1, 1], color=PRIMARY)
    sns.kdeplot(final_hybrid_scores,   label="Hybrid Score", fill=True, alpha=0.35, ax=axes4[1, 1], color=ACCENT)
    axes4[1, 1].axvline(DEFAULT_THRESHOLD_HYBRID, color=DANGER, linestyle="--", linewidth=2,
                        label=f"Threshold {DEFAULT_THRESHOLD_HYBRID}")
    style_axis(axes4[1, 1], "Score Shift", "Score", "Density")
    axes4[1, 1].legend()

    axes4[1, 2].axis("off")
    axes4[1, 2].text(0.02, 0.98,
        f"Manual FIS Accuracy : {format_pct(acc_manual)}\n"
        f"Neuro-Fuzzy Accuracy: {format_pct(acc_hybrid)}\n"
        f"Improvement         : {format_pp((acc_hybrid - acc_manual)*100)}\n\n"
        f"Train Data          : {len(X_train_ann)}\n"
        f"Test Data           : {len(X_test_ann)}\n"
        f"ANN Epoch           : 300\n"
        f"Final Threshold     : {DEFAULT_THRESHOLD_HYBRID}\n\n"
        f"Hybrid Rule:\n"
        f"- FIS uncertain (30-70) → ANN weighted more\n"
        f"- FIS confident → FIS dominant",
        transform=axes4[1, 2].transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor=BORDER))
    axes4[1, 2].set_title("Result Summary", fontweight="bold", color=SECONDARY)
    plt.tight_layout()
    return fig4


# ============================================================
# HELPER: result box
# ============================================================
def _render_model_result_box(model_name, model_tag, pred, score, threshold, is_loading=False):
    if is_loading:
        st.markdown(f"""
            <div class="result-loading">
                <div class="model-verdict-label">{model_name}</div>
                <div style="font-size:0.9rem;color:{TEXT_MUTED};margin-top:4px;">{model_tag}</div>
            </div>""", unsafe_allow_html=True)
        return

    approved      = pred == 1
    box_class     = "result-approved" if approved else "result-rejected"
    verdict_text  = "DISETUJUI" if approved else "DITOLAK"
    verdict_color = "#065F46" if approved else "#991B1B"
    bar_color     = ACCENT if approved else DANGER
    bar_w         = min(float(score), 100)

    st.markdown(f"""
        <div class="{box_class}">
            <div class="model-verdict-label">{model_name}</div>
            <div style="font-size:1.3rem;font-weight:800;color:{verdict_color};margin-bottom:4px;">{verdict_text}</div>
            <div style="font-size:0.88rem;color:{TEXT_MUTED};margin-top:2px;margin-bottom:14px;">{model_tag}</div>
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:{TEXT_MUTED};margin-bottom:4px;">
                <span>Skor</span>
                <span style="font-weight:700;color:{SECONDARY};">{score:.1f} / 100</span>
            </div>
            <div class="score-bar-wrap">
                <div style="height:100%;border-radius:999px;background:{bar_color};width:{bar_w:.1f}%;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:{TEXT_MUTED};">
                <span>0</span><span>Threshold: {threshold}</span><span>100</span>
            </div>
        </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE: PREDIKSI KELAYAKAN
# ============================================================
def render_predict_page(p):
    st.markdown(f"""
        <div class="page-title">
            <div class="page-title-main">Prediksi Kelayakan Kredit</div>
            <div class="page-title-sub">
                Masukkan data pemohon untuk prediksi dari tiga model:
                Manual FIS, GA-Optimized FIS, dan Neuro-Fuzzy ANN Hybrid.
            </div>
        </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="input-section-title">Informasi Keuangan</div>', unsafe_allow_html=True)
        applicant_income   = st.number_input("Pendapatan Pemohon (Rp / bulan)",
            min_value=0, max_value=100_000_000, value=5_000_000, step=100_000, format="%d")
        coapplicant_income = st.number_input("Pendapatan Co-Applicant (Rp / bulan)",
            min_value=0, max_value=100_000_000, value=0, step=100_000, format="%d")
        loan_amount        = st.number_input("Jumlah Pinjaman (ribuan Rp)",
            min_value=1, max_value=700, value=150, step=1)
        st.markdown('<div class="input-section-title">Informasi Pemohon</div>', unsafe_allow_html=True)
        credit_history = st.radio("Riwayat Kredit", options=[1, 0],
            format_func=lambda x: "Baik — tidak ada kredit macet" if x == 1 else "Buruk — ada riwayat kredit macet")

    with col_right:
        st.markdown('<div class="input-section-title">Informasi Tambahan</div>', unsafe_allow_html=True)
        gender        = st.selectbox("Jenis Kelamin",          ["Male", "Female"])
        married       = st.selectbox("Status Pernikahan",      ["Yes", "No"])
        dependents    = st.selectbox("Jumlah Tanggungan",      ["0", "1", "2", "3+"])
        education     = st.selectbox("Pendidikan",             ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Wiraswasta",             ["No", "Yes"])
        property_area = st.selectbox("Area Properti",          ["Urban", "Semiurban", "Rural"])
        loan_term     = st.selectbox("Tenor Pinjaman (bulan)", [360, 120, 180, 240, 300, 480])

        total_income = applicant_income + coapplicant_income
        income_ratio = (total_income / loan_amount) if loan_amount > 0 else 100
        st.markdown(f"""
            <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;padding:14px 16px;margin-top:14px;">
                <div style="font-size:0.75rem;color:{TEXT_MUTED};font-weight:600;text-transform:uppercase;">Preview Rasio</div>
                <div style="margin-top:8px;display:flex;gap:24px;">
                    <div>
                        <div style="font-size:0.72rem;color:{TEXT_MUTED};">Total Income</div>
                        <div style="font-size:1rem;font-weight:800;color:{SECONDARY};">Rp {total_income:,.0f}</div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem;color:{TEXT_MUTED};">Income / Loan Ratio</div>
                        <div style="font-size:1rem;font-weight:800;color:{PRIMARY};">{income_ratio:.1f}x</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if not st.button("Prediksi Kelayakan", type="primary", use_container_width=True):
        return

    st.markdown(f"""<br>
        <div style="font-size:1.1rem;font-weight:800;color:{SECONDARY};margin-bottom:14px;">
            Hasil Prediksi — Tiga Model
        </div>""", unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3, gap="medium")

    # Manual FIS
    pred_fis, score_fis, activations = p["fis"].predict_single(
        applicant_income, loan_amount, credit_history, coapplicant_income)
    with mc1:
        _render_model_result_box("Manual FIS",
            f"Rule-based Fuzzy · Threshold {DEFAULT_THRESHOLD_FIS}",
            pred_fis, score_fis, DEFAULT_THRESHOLD_FIS)

    # GA FIS
    with mc2:
        try:
            pred_ga, score_ga, _ = p["ga_fis"].predict_single(
                applicant_income, loan_amount, credit_history, coapplicant_income)
            _render_model_result_box("GA-Optimized FIS",
                f"Genetic Algorithm · Threshold {p['threshold_ga']:.1f}",
                pred_ga, score_ga, f"{p['threshold_ga']:.1f}")
        except Exception as e:
            _render_model_result_box("GA-Optimized FIS", f"Error: {e}",
                                     None, None, None, is_loading=True)

    # ANN Hybrid
    with mc3:
        try:
            sample = np.array([[applicant_income, coapplicant_income, loan_amount, credit_history]], dtype=float)
            sample_scaled = p["ann_scaler"].transform(sample)
            with torch.no_grad():
                ann_prob = p["ann_model"](torch.FloatTensor(sample_scaled)).item() * 100
            if 30 < score_fis < 70:
                hybrid_score = score_fis * 0.3 + ann_prob * 0.7
            else:
                hybrid_score = score_fis * 0.8 + ann_prob * 0.2
            pred_ann = 1 if hybrid_score >= DEFAULT_THRESHOLD_HYBRID else 0
            _render_model_result_box("Neuro-Fuzzy ANN",
                f"Hybrid FIS+ANN · Threshold {DEFAULT_THRESHOLD_HYBRID}",
                pred_ann, hybrid_score, DEFAULT_THRESHOLD_HYBRID)
        except Exception as e:
            _render_model_result_box("Neuro-Fuzzy ANN", f"Error: {e}",
                                     None, None, None, is_loading=True)

    # Analisis Faktor
    st.markdown(f"""<br>
        <div style="font-size:1rem;font-weight:700;color:{SECONDARY};margin-bottom:10px;">Analisis Faktor</div>""",
        unsafe_allow_html=True)
    fc1, fc2 = st.columns([1, 1], gap="large")

    with fc1:
        ch_s,  ch_c  = ("Baik",  ACCENT)  if credit_history == 1 else ("Buruk", DANGER)
        inc_s, inc_c = (("Tinggi", ACCENT) if total_income >= 8_000_000 else
                        ("Sedang", WARNING) if total_income >= 4_000_000 else ("Rendah", DANGER))
        rat_s, rat_c = (("Sangat Baik", ACCENT) if income_ratio >= 60 else
                        ("Cukup", WARNING) if income_ratio >= 20 else ("Kurang", DANGER))
        ln_s,  ln_c  = (("Kecil", ACCENT) if loan_amount <= 100 else
                        ("Sedang", WARNING) if loan_amount <= 250 else ("Besar", DANGER))
        for name, val, color, hint in [
            ("Riwayat Kredit",     ch_s,                                    ch_c,  "Faktor paling berpengaruh"),
            ("Total Pendapatan",   f"Rp {total_income:,.0f} — {inc_s}",     inc_c, "Pendapatan gabungan"),
            ("Rasio Income/Loan",  f"{income_ratio:.1f}x — {rat_s}",        rat_c, "Semakin tinggi semakin baik"),
            ("Jumlah Pinjaman",    f"{loan_amount} (ribuan) — {ln_s}",      ln_c,  "Relatif terhadap kemampuan bayar"),
        ]:
            st.markdown(f"""
                <div class="factor-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <div style="font-size:0.85rem;font-weight:700;color:{SECONDARY};">{name}</div>
                            <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:2px;">{hint}</div>
                        </div>
                        <div style="font-size:0.82rem;font-weight:700;color:{color};text-align:right;max-width:180px;">{val}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

    with fc2:
        st.markdown(f'<div style="font-size:0.85rem;font-weight:700;color:{SECONDARY};margin-bottom:10px;">Aktivasi Fuzzy Rule</div>',
                    unsafe_allow_html=True)
        for act_label, act_val, act_color in [
            ("Reject",  activations.get("reject", 0),  DANGER),
            ("Review",  activations.get("review", 0),  WARNING),
            ("Approve", activations.get("approve", 0), ACCENT),
        ]:
            st.markdown(f"""
                <div class="metric-card" style="border-left:4px solid {act_color};margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div class="metric-label" style="margin-bottom:0;">{act_label}</div>
                        <div style="font-size:1rem;font-weight:800;color:{act_color};">{act_val:.3f}</div>
                    </div>
                    <div class="score-bar-wrap" style="margin:6px 0 0 0;">
                        <div style="height:100%;border-radius:999px;background:{act_color};width:{act_val*100:.1f}%;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        bar_color_fis = ACCENT if pred_fis == 1 else DANGER
        st.markdown(f"""
            <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;padding:12px 14px;margin-top:4px;">
                <div style="font-size:0.75rem;color:{TEXT_MUTED};font-weight:600;">Skor Defuzzifikasi (Manual FIS)</div>
                <div style="font-size:1.4rem;font-weight:800;color:{SECONDARY};margin:4px 0;">{score_fis:.2f} / 100</div>
                <div class="score-bar-wrap">
                    <div style="height:100%;border-radius:999px;background:{bar_color_fis};width:{min(score_fis,100):.1f}%;"></div>
                </div>
                <div style="font-size:0.72rem;color:{TEXT_MUTED};">Threshold Manual FIS: {DEFAULT_THRESHOLD_FIS}</div>
            </div>""", unsafe_allow_html=True)

    with st.expander("Lihat Ringkasan Data Input"):
        st.dataframe(pd.DataFrame({
            "Parameter": ["Jenis Kelamin","Status Menikah","Tanggungan","Pendidikan","Wiraswasta",
                          "Pendapatan Pemohon","Pendapatan Co-Applicant","Total Pendapatan",
                          "Jumlah Pinjaman","Tenor","Riwayat Kredit","Area Properti",
                          "Rasio Income/Loan","Skor FIS","Hasil FIS"],
            "Nilai": [gender, married, dependents, education, self_employed,
                      f"Rp {applicant_income:,.0f}", f"Rp {coapplicant_income:,.0f}",
                      f"Rp {total_income:,.0f}", f"{loan_amount} (ribuan)", f"{loan_term} bulan",
                      "Baik (1)" if credit_history == 1 else "Buruk (0)", property_area,
                      f"{income_ratio:.2f}x", f"{score_fis:.2f} / 100",
                      "DISETUJUI" if pred_fis == 1 else "DITOLAK"],
        }), use_container_width=True, hide_index=True)


# ============================================================
# PAGE: ANALISIS & TRAINING
# ============================================================
def render_analysis_page(p):
    m = p["metrics"]
    st.markdown(f"""
        <div class="page-title">
            <div class="page-title-main">Analisis & Training Model</div>
            <div class="page-title-sub">
                Hasil pipeline dari notebook UTS_SoftComp — dimuat langsung dari file export,
                tanpa training ulang.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background:linear-gradient(135deg,#ECFDF5 0%,#D1FAE5 100%);
                    border:2px solid #10B981;border-radius:14px;padding:14px 20px;margin-bottom:20px;">
            <span style="font-size:1rem;font-weight:700;color:#065F46;">
                ✅ Semua model dimuat dari file export notebook — tidak ada training ulang
            </span>
        </div>""", unsafe_allow_html=True)

    # Dataset summary
    section_header("Dataset", "Ringkasan Dataset", "Sumber: data_train.csv (via notebook)")
    col_a, col_b, col_c = st.columns(3)
    with col_a: metric_card("Total Rows",  f"{int(m['n_rows']):,}",     "Baris valid")
    with col_b: metric_card("Approved",    f"{int(m['n_approved']):,}", "Label approval")
    with col_c: metric_card("Rejected",    f"{int(m['n_rejected']):,}", "Label rejection")

    if p["df"] is not None:
        st.dataframe(p["df"].head(10), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(
        ["Tahap 1 — Manual FIS", "Tahap 2 — GA Optimization", "Tahap 3 — ANN Hybrid"])

    # ── TAB 1 ──────────────────────────────────────────────────────────────
    with tab1:
        section_header("Tahap 1", "Manual Fuzzy Inference System",
                        "Baseline model berbasis rule fuzzy dan membership function manual.")
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Accuracy",     format_pct(m["acc_fis"]),    "Baseline accuracy")
        with c2: metric_card("Fuzzy Rules",  "11",                         "Total rule")
        with c3: metric_card("Test Samples", f"{int(m['n_test_fis'])}",   "Data evaluasi")

        st.subheader("Membership Functions")
        st.pyplot(p["fig_mf_fis"], use_container_width=True)
        st.subheader("Evaluation Result")
        st.pyplot(p["fig_eval_fis"], use_container_width=True)

    # ── TAB 2 ──────────────────────────────────────────────────────────────
    with tab2:
        section_header("Tahap 2", "Genetic Algorithm Optimization",
                        "Optimasi MF + threshold menggunakan GA "
                        "(fitness: 0.4×BalAcc + 0.6×F1_macro, dynamic mutation rate).")

        s1, s2, s3 = st.columns(3)
        with s1: metric_card("Train Split",      f"{int(m['n_train_ga'])}",  "Data training")
        with s2: metric_card("Validation Split", f"{int(m['n_val_ga'])}",    "Fitness GA")
        with s3: metric_card("Test Split",       f"{int(m['n_test_ga'])}",   "Blind evaluation")

        st.caption(
            f"Baseline Manual FIS — val: acc={m['acc_val_bl']:.4f}  f1={m['f1_val_bl']:.4f}  "
            f"fitness={m['baseline_fitness']:.4f} | "
            f"test: acc={m['acc_test_bl']:.4f}  f1={m['f1_test_bl']:.4f}")

        st.subheader("GA Convergence")
        st.pyplot(p["fig_konvergensi"], use_container_width=True)

        st.subheader("Parameter Comparison (Manual vs GA-Optimized)")
        manual_mf = p["fis"].mf_params
        ga_mf     = chromosome_to_params(p["best_chrom"])
        rows = []
        for var in STRUCTURE:
            for label in STRUCTURE[var]:
                rows.append({
                    "Variable": var, "Label": label,
                    "Manual":   str([round(v, 2) for v in manual_mf[var][label]]),
                    "GA Optimized": str([round(v, 2) for v in ga_mf[var][label]]),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        thr1, thr2 = st.columns(2)
        with thr1: metric_card("Threshold Manual", "52.00", "Nilai statis")
        with thr2: metric_card("Threshold GA",     f"{m['best_threshold_ga']:.2f}",
                               f"Best fitness val: {m['best_fitness_ga']:.4f}")

        g1, g2, g3 = st.columns(3)
        with g1: metric_card("Manual Accuracy (test)", format_pct(m["acc_test_bl"]),  "Baseline test GA")
        with g2: metric_card("GA Accuracy (test)",     format_pct(m["acc_ga"]),       "Setelah optimasi")
        with g3: metric_card("Delta Accuracy",         format_pp(m["improvement_pp_ga"]), "Perubahan performa")

        g4, g5, g6 = st.columns(3)
        with g4: metric_card("Manual F1 Macro", format_pct(m["f1_test_bl"]), "Baseline test")
        with g5: metric_card("GA F1 Macro",     format_pct(m["f1_ga"]),      "Setelah optimasi")
        with g6: metric_card("GA Balanced Acc", format_pct(m["bal_acc_ga"]), "Robust imbalance")

        st.subheader("Final GA Evaluation")
        st.pyplot(p["fig_ga_eval"], use_container_width=True)
        st.subheader("Membership Function Comparison")
        st.pyplot(p["fig_mf_ga"], use_container_width=True)

        st.subheader("Ringkasan Tahap 2")
        st.markdown(f"""
            <div class="section-card">
                <table style="width:100%;font-size:0.95rem;border-collapse:collapse;">
                    <tr><td style="padding:4px 8px;color:{TEXT_MUTED};">Manual FIS val acc</td>
                        <td style="padding:4px 8px;font-weight:700;">{m['acc_val_bl']:.4f}</td></tr>
                    <tr><td style="padding:4px 8px;color:{TEXT_MUTED};">Manual FIS test acc</td>
                        <td style="padding:4px 8px;font-weight:700;">{m['acc_test_bl']:.4f}</td></tr>
                    <tr><td style="padding:4px 8px;color:{TEXT_MUTED};">GA-Optimized test acc</td>
                        <td style="padding:4px 8px;font-weight:700;">{m['acc_ga']:.4f}</td></tr>
                    <tr><td style="padding:4px 8px;color:{TEXT_MUTED};">GA F1 Macro</td>
                        <td style="padding:4px 8px;font-weight:700;">{m['f1_ga']:.4f}</td></tr>
                    <tr><td style="padding:4px 8px;color:{TEXT_MUTED};">Delta Accuracy</td>
                        <td style="padding:4px 8px;font-weight:700;">{m['improvement_pp_ga']:+.2f} pp</td></tr>
                    <tr><td style="padding:4px 8px;color:{TEXT_MUTED};">Threshold GA</td>
                        <td style="padding:4px 8px;font-weight:700;">{m['best_threshold_ga']:.2f} (Manual: 52)</td></tr>
                </table>
            </div>""", unsafe_allow_html=True)

    # ── TAB 3 ──────────────────────────────────────────────────────────────
    with tab3:
        section_header("Tahap 3", "Neuro-Fuzzy ANN Hybrid",
                        "Hybrid model yang menggabungkan output FIS dengan prediksi ANN.")
        a1, a2, a3 = st.columns(3)
        with a1: metric_card("Manual FIS",   format_pct(m["acc_manual_ann"]),    "Akurasi baseline")
        with a2: metric_card("Neuro-Fuzzy",  format_pct(m["acc_hybrid_ann"]),    "Akurasi hybrid")
        with a3: metric_card("Improvement",  format_pp(m["improvement_pp_ann"]), "Vs baseline")

        st.subheader("Integrated Evaluation Dashboard")
        st.pyplot(p["fig_ann"], use_container_width=True)

    st.success("Semua model dimuat dari export notebook. Siap digunakan di halaman Prediksi.")


# ============================================================
# MAIN
# ============================================================
def main():
    try:
        pipeline = load_pipeline()
    except FileNotFoundError as e:
        st.error(str(e))
        st.markdown("""
        **Langkah-langkah:**
        1. Buka notebook `UTS_SoftComp_(5).ipynb` di Google Colab
        2. Jalankan semua cell dari atas
        3. Jalankan cell **EXPORT** paling bawah — akan muncul dialog download
        4. Download semua 5 file: `ga_best_chrom.npz`, `ann_model.pt`, `scaler.pkl`,
           `pipeline_metrics.npz`, `ga_history.npz`
        5. Letakkan semua file tersebut + `data_train.csv` di folder yang sama dengan `app.py`
        6. Jalankan ulang: `streamlit run app.py`
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Gagal memuat pipeline: {e}")
        st.stop()

    if st.session_state.active_page == "predict":
        render_predict_page(pipeline)
    elif st.session_state.active_page == "analysis":
        render_analysis_page(pipeline)

main()