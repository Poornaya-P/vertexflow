"""
╔══════════════════════════════════════════════════════════════════╗
║              ⚡ VertexFlow — ML Platform (single file)           ║
║                                                                  ║
║  Required modules:                                               ║
║    pip install streamlit scikit-learn pandas numpy plotly        ║
║                optuna xgboost lightgbm openpyxl                  ║
║                                                                  ║
║  Run:  streamlit run vertexflow.py                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime

# ── Sklearn datasets ──────────────────────────────────────────────
from sklearn.datasets import (
    make_classification, make_regression, make_blobs,
    load_iris, load_breast_cancer, load_wine, load_diabetes,
    fetch_california_housing,
)
# ── Preprocessing ─────────────────────────────────────────────────
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, KFold, GridSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
# ── Metrics ───────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, log_loss,
    precision_recall_curve, roc_curve, average_precision_score,
)
from sklearn.inspection import permutation_importance
# ── Models ────────────────────────────────────────────────────────
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    BaggingClassifier,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ── Optional boosting ─────────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ── Optional Optuna ───────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ── Plotly ────────────────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VertexFlow — ML Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0d1117; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1117 0%,#161b22 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stMetric"] {
    background:#161b22; border:1px solid #21262d;
    border-radius:12px; padding:16px !important;
}
[data-testid="stMetricValue"] {
    font-family:'Space Mono',monospace; color:#58a6ff; font-size:1.5rem !important;
}
[data-testid="stMetricLabel"] { color:#8b949e; font-size:11px; text-transform:uppercase; letter-spacing:.08em; }

.stButton > button {
    background:linear-gradient(135deg,#1f6feb,#388bfd);
    color:white; border:none; border-radius:8px; font-weight:600;
    transition:all .2s;
}
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 4px 20px #1f6feb55; }

.stTabs [data-baseweb="tab-list"] { background:#161b22; border-radius:10px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { border-radius:8px; color:#8b949e; font-weight:500; }
.stTabs [aria-selected="true"] { background:#1f6feb22 !important; color:#58a6ff !important; }

.section-header {
    font-family:'Space Mono',monospace; color:#58a6ff; font-size:12px;
    text-transform:uppercase; letter-spacing:.12em;
    border-left:3px solid #1f6feb; padding-left:10px; margin:20px 0 12px 0;
}
.info-card {
    background:#161b22; border:1px solid #21262d;
    border-radius:12px; padding:20px; margin:8px 0;
}
.success-card { border-left:4px solid #3fb950; }
.accent-card  { border-left:4px solid #58a6ff; }
.warn-card    { border-left:4px solid #d29922; }

.hero-title {
    font-size:2.6rem; font-weight:700;
    background:linear-gradient(135deg,#58a6ff 0%,#a371f7 50%,#3fb950 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.1;
}
.hero-sub { color:#8b949e; font-size:1.05rem; font-weight:400; margin-top:8px; }
hr { border-color:#21262d; }
.stProgress > div > div { background:linear-gradient(90deg,#1f6feb,#58a6ff); }
.stSelectbox > div > div, .stMultiSelect > div > div {
    background:#161b22 !important; border:1px solid #21262d !important; border-radius:8px !important;
}
.badge {
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-size:11px; font-weight:600; font-family:'Space Mono',monospace; letter-spacing:.05em;
}
.badge-blue   { background:#1f6feb22; color:#58a6ff; border:1px solid #1f6feb44; }
.badge-green  { background:#3fb95022; color:#3fb950; border:1px solid #3fb95044; }
.badge-purple { background:#a371f722; color:#a371f7; border:1px solid #a371f744; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# THEME CONSTANTS
# ══════════════════════════════════════════════════════════════════
BG      = "#0d1117"
SURFACE = "#161b22"
BORDER  = "#21262d"
BLUE    = "#58a6ff"
GREEN   = "#3fb950"
YELLOW  = "#d29922"
RED     = "#f85149"
PURPLE  = "#a371f7"
MUTED   = "#8b949e"
TEXT    = "#c9d1d9"

LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=SURFACE,
    font=dict(family="Inter, sans-serif", color=TEXT, size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1),
)
GRID = dict(
    xaxis=dict(gridcolor=BORDER, gridwidth=1, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, gridwidth=1, zerolinecolor=BORDER),
)

def _fig(fig, title="", height=400):
    fig.update_layout(**LAYOUT, **GRID,
                      title=dict(text=title, font_size=14), height=height)
    return fig


# ══════════════════════════════════════════════════════════════════
# ── SECTION 1: DATA UTILITIES ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

BUILTIN_DATASETS = {
    "Iris (multiclass)"         : "iris",
    "Breast Cancer (binary)"    : "breast_cancer",
    "Wine Quality (multiclass)" : "wine",
    "Diabetes (regression)"     : "diabetes",
    "California Housing (reg.)" : "california",
}

SYNTHETIC_TEMPLATES = {
    "Binary Classification"     : "binary_clf",
    "Multiclass (3 classes)"    : "multi_clf",
    "Regression (linear)"       : "regression",
    "Imbalanced Classification" : "imbalanced_clf",
    "Clustered / Non-linear"    : "nonlinear_clf",
}

DS_DESC = {
    "Iris (multiclass)"         : "150 flowers · 4 features · 3 species. Classic benchmark.",
    "Breast Cancer (binary)"    : "569 samples · 30 features. Malignant vs. benign tumour.",
    "Wine Quality (multiclass)" : "178 wines · 13 chemical features · 3 quality classes.",
    "Diabetes (regression)"     : "442 patients · 10 features. Predict disease progression.",
    "California Housing (reg.)" : "20,640 blocks. Predict median house value.",
}


def load_builtin(name):
    key = BUILTIN_DATASETS[name]
    if key == "iris":           d, task = load_iris(as_frame=True),             "multiclass"
    elif key == "breast_cancer":d, task = load_breast_cancer(as_frame=True),    "binary"
    elif key == "wine":         d, task = load_wine(as_frame=True),             "multiclass"
    elif key == "diabetes":     d, task = load_diabetes(as_frame=True),         "regression"
    else:                       d, task = fetch_california_housing(as_frame=True), "regression"
    df = d.frame.copy()
    tc = d.target.name if hasattr(d.target, "name") else "target"
    if tc not in df.columns: df[tc] = d.target.values
    return df, tc, task


def gen_synthetic(template, n=500, nf=8, noise=0.15, seed=42):
    if template == "binary_clf":
        X, y = make_classification(n, nf, n_informative=max(2,nf//2),
                                   n_redundant=2, flip_y=noise, random_state=seed)
        task = "binary"
    elif template == "multi_clf":
        X, y = make_classification(n, nf, n_informative=max(3,nf//2), n_redundant=2,
                                   n_classes=3, n_clusters_per_class=1,
                                   flip_y=noise, random_state=seed)
        task = "multiclass"
    elif template == "regression":
        X, y = make_regression(n, nf, n_informative=max(3,nf//2),
                               noise=noise*50, random_state=seed)
        task = "regression"
    elif template == "imbalanced_clf":
        X, y = make_classification(n, nf, n_informative=max(2,nf//2), n_redundant=2,
                                   weights=[0.85,0.15], flip_y=0.01, random_state=seed)
        task = "binary"
    else:
        X, y = make_blobs(n, centers=3, n_features=min(nf,10),
                          cluster_std=1.5+noise*5, random_state=seed)
        y = (y > 0).astype(int); task = "binary"

    cols = [f"feature_{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols); df["target"] = y
    return df, "target", task


def auto_task(df, tc):
    s = df[tc]; n = s.nunique()
    if n == 2: return "binary"
    if n <= 20 and (s.dtype == object or n < 15): return "multiclass"
    return "regression"


def preprocess(df, tc, scale=True):
    df = df.copy().dropna()
    for col in df.select_dtypes("object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    feats = [c for c in df.columns if c != tc]
    X = df[feats].values.astype(float); y = df[tc].values
    strat = y if len(np.unique(y)) < 50 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=strat)
    sc = None
    if scale:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    return X_tr, X_te, y_tr, y_te, feats, sc


def data_quality(df):
    total = len(df)
    score = 100.0 - df.isnull().mean().mean()*200 - df.duplicated().mean()*150
    cols  = []
    for col in df.columns:
        s = df[col]
        cols.append({
            "Column": col, "Type": str(s.dtype),
            "Missing": int(s.isnull().sum()),
            "Missing%": round(s.isnull().mean()*100,2),
            "Unique": int(s.nunique()),
            "Mean": round(float(s.mean()),4) if pd.api.types.is_numeric_dtype(s) else "—",
            "Std":  round(float(s.std()),4)  if pd.api.types.is_numeric_dtype(s) else "—",
            "Min":  round(float(s.min()),4)  if pd.api.types.is_numeric_dtype(s) else "—",
            "Max":  round(float(s.max()),4)  if pd.api.types.is_numeric_dtype(s) else "—",
        })
    return max(0, round(score,1)), pd.DataFrame(cols)


# ══════════════════════════════════════════════════════════════════
# ── SECTION 2: METRICS ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def clf_metrics(y_true, y_pred, y_proba=None, task="binary"):
    avg = "binary" if task == "binary" else "weighted"
    m = {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall"   : recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1"       : f1_score(y_true, y_pred, average=avg, zero_division=0),
        "f1_macro" : f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro" : f1_score(y_true, y_pred, average="micro", zero_division=0),
        "conf_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report"   : classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    if y_proba is not None:
        try:
            if task == "binary":
                p = y_proba[:,1] if y_proba.ndim==2 else y_proba
                m["roc_auc"]       = roc_auc_score(y_true, p)
                m["avg_precision"] = average_precision_score(y_true, p)
                m["log_loss"]      = log_loss(y_true, y_proba)
                fpr,tpr,_ = roc_curve(y_true, p)
                m["roc_curve"]     = {"fpr":fpr.tolist(),"tpr":tpr.tolist()}
                pr,rc,_   = precision_recall_curve(y_true, p)
                m["pr_curve"]      = {"precision":pr.tolist(),"recall":rc.tolist()}
            else:
                m["roc_auc"]  = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                m["log_loss"] = log_loss(y_true, y_proba)
        except Exception:
            pass
    return m


def reg_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)*100
    ss_res = np.sum((y_true-y_pred)**2)
    ss_tot = np.sum((y_true-np.mean(y_true))**2)
    return {
        "mae":mae, "mse":mse, "rmse":rmse, "r2":r2,
        "mape":mape, "explained_var":1-ss_res/(ss_tot+1e-9),
        "y_pred":y_pred.tolist(), "y_true":y_true.tolist(),
    }


def feat_importance(model, X_te, y_te, feats, task):
    imp, method = None, None
    if hasattr(model, "feature_importances_"):
        imp, method = model.feature_importances_, "model"
    elif hasattr(model, "coef_"):
        c = model.coef_
        imp, method = (np.abs(c).mean(axis=0) if c.ndim>1 else np.abs(c)), "coefficient"
    if imp is None:
        try:
            sc = "accuracy" if task!="regression" else "r2"
            r  = permutation_importance(model, X_te, y_te, n_repeats=8, random_state=42, scoring=sc)
            imp, method = r.importances_mean, "permutation"
        except Exception:
            imp, method = np.ones(len(feats))/len(feats), "uniform"
    imp = np.array(imp)
    if imp.sum(): imp = imp/imp.sum()
    return pd.DataFrame({"feature":feats,"importance":imp,"method":method})\
             .sort_values("importance",ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# ── SECTION 3: MODEL CATALOG ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

CLF_MODELS = {
    "Logistic Regression"    : {"cls":LogisticRegression,       "params":{"max_iter":1000,"random_state":42},   "family":"Linear"},
    "LDA"                    : {"cls":LinearDiscriminantAnalysis,"params":{},                                   "family":"Linear"},
    "SGD Classifier"         : {"cls":SGDClassifier,            "params":{"max_iter":1000,"random_state":42},   "family":"Linear"},
    "Gaussian Naive Bayes"   : {"cls":GaussianNB,               "params":{},                                   "family":"Bayesian"},
    "Decision Tree"          : {"cls":DecisionTreeClassifier,   "params":{"random_state":42},                  "family":"Tree"},
    "Random Forest"          : {"cls":RandomForestClassifier,   "params":{"n_estimators":200,"random_state":42,"n_jobs":-1}, "family":"Ensemble"},
    "Extra Trees"            : {"cls":ExtraTreesClassifier,     "params":{"n_estimators":200,"random_state":42,"n_jobs":-1}, "family":"Ensemble"},
    "Gradient Boosting"      : {"cls":GradientBoostingClassifier,"params":{"n_estimators":200,"random_state":42},"family":"Ensemble"},
    "AdaBoost"               : {"cls":AdaBoostClassifier,       "params":{"n_estimators":100,"random_state":42},"family":"Ensemble"},
    "Bagging"                : {"cls":BaggingClassifier,        "params":{"n_estimators":50,"random_state":42}, "family":"Ensemble"},
    "SVM (RBF)"              : {"cls":SVC,                      "params":{"kernel":"rbf","probability":True,"random_state":42}, "family":"SVM"},
    "SVM (Linear)"           : {"cls":SVC,                      "params":{"kernel":"linear","probability":True,"random_state":42}, "family":"SVM"},
    "K-Nearest Neighbors"    : {"cls":KNeighborsClassifier,     "params":{"n_neighbors":5},                    "family":"Neighbors"},
    "MLP Neural Network"     : {"cls":MLPClassifier,            "params":{"hidden_layer_sizes":(128,64),"max_iter":500,"early_stopping":True,"random_state":42}, "family":"Neural Net"},
}
if HAS_XGB:
    CLF_MODELS["XGBoost"] = {"cls":XGBClassifier, "params":{"n_estimators":200,"random_state":42,"eval_metric":"logloss","verbosity":0}, "family":"Boosting"}
if HAS_LGB:
    CLF_MODELS["LightGBM"] = {"cls":LGBMClassifier, "params":{"n_estimators":200,"random_state":42,"verbose":-1}, "family":"Boosting"}

REG_MODELS = {
    "Linear Regression"      : {"cls":LinearRegression,         "params":{},                                   "family":"Linear"},
    "Ridge"                  : {"cls":Ridge,                    "params":{"alpha":1.0},                        "family":"Linear"},
    "Lasso"                  : {"cls":Lasso,                    "params":{"alpha":0.1,"max_iter":5000},        "family":"Linear"},
    "ElasticNet"             : {"cls":ElasticNet,               "params":{"alpha":0.1,"l1_ratio":0.5,"max_iter":5000}, "family":"Linear"},
    "Decision Tree (Reg.)"   : {"cls":DecisionTreeRegressor,    "params":{"random_state":42},                  "family":"Tree"},
    "Random Forest (Reg.)"   : {"cls":RandomForestRegressor,    "params":{"n_estimators":200,"random_state":42,"n_jobs":-1}, "family":"Ensemble"},
    "Gradient Boosting (Reg.)":{"cls":GradientBoostingRegressor,"params":{"n_estimators":200,"random_state":42},"family":"Ensemble"},
    "Extra Trees (Reg.)"     : {"cls":ExtraTreesRegressor,      "params":{"n_estimators":200,"random_state":42,"n_jobs":-1}, "family":"Ensemble"},
    "AdaBoost (Reg.)"        : {"cls":AdaBoostRegressor,        "params":{"n_estimators":100,"random_state":42},"family":"Ensemble"},
    "SVR (RBF)"              : {"cls":SVR,                      "params":{"kernel":"rbf"},                     "family":"SVM"},
    "KNN Regressor"          : {"cls":KNeighborsRegressor,      "params":{"n_neighbors":5},                    "family":"Neighbors"},
    "MLP Regressor"          : {"cls":MLPRegressor,             "params":{"hidden_layer_sizes":(128,64),"max_iter":500,"early_stopping":True,"random_state":42}, "family":"Neural Net"},
}
if HAS_XGB:
    REG_MODELS["XGBoost (Reg.)"] = {"cls":XGBRegressor, "params":{"n_estimators":200,"random_state":42,"verbosity":0}, "family":"Boosting"}
if HAS_LGB:
    REG_MODELS["LightGBM (Reg.)"] = {"cls":LGBMRegressor, "params":{"n_estimators":200,"random_state":42,"verbose":-1}, "family":"Boosting"}


def optuna_space(trial, name, task):
    """Return model with Optuna-suggested hyperparameters."""
    if "Random Forest" in name:
        p = {"n_estimators":trial.suggest_int("n_estimators",50,500),
             "max_depth":trial.suggest_int("max_depth",3,20),
             "min_samples_split":trial.suggest_int("min_samples_split",2,20),
             "min_samples_leaf":trial.suggest_int("min_samples_leaf",1,10),
             "max_features":trial.suggest_categorical("max_features",["sqrt","log2",None]),
             "random_state":42}
        cls = RandomForestRegressor if task=="regression" else RandomForestClassifier

    elif "Gradient Boosting" in name:
        p = {"n_estimators":trial.suggest_int("n_estimators",50,500),
             "learning_rate":trial.suggest_float("learning_rate",0.01,0.3,log=True),
             "max_depth":trial.suggest_int("max_depth",2,8),
             "subsample":trial.suggest_float("subsample",0.5,1.0),
             "min_samples_split":trial.suggest_int("min_samples_split",2,20),
             "random_state":42}
        cls = GradientBoostingRegressor if task=="regression" else GradientBoostingClassifier

    elif name == "Logistic Regression":
        p = {"C":trial.suggest_float("C",1e-4,100,log=True),
             "penalty":trial.suggest_categorical("penalty",["l1","l2"]),
             "solver":"saga","max_iter":2000,"random_state":42}
        cls = LogisticRegression

    elif "SVM" in name:
        p = {"C":trial.suggest_float("C",1e-3,100,log=True),
             "gamma":trial.suggest_categorical("gamma",["scale","auto"]),
             "probability":True,"random_state":42}
        cls = SVR if task=="regression" else SVC

    elif "KNN" in name or "K-Nearest" in name:
        p = {"n_neighbors":trial.suggest_int("n_neighbors",1,30),
             "weights":trial.suggest_categorical("weights",["uniform","distance"])}
        cls = KNeighborsRegressor if task=="regression" else KNeighborsClassifier

    elif "MLP" in name:
        sz  = trial.suggest_int("layer_size",32,256)
        nl  = trial.suggest_int("n_layers",1,3)
        p = {"hidden_layer_sizes":tuple([sz]*nl),
             "alpha":trial.suggest_float("alpha",1e-5,1e-1,log=True),
             "learning_rate_init":trial.suggest_float("lr",1e-4,1e-2,log=True),
             "max_iter":500,"early_stopping":True,"random_state":42}
        cls = MLPRegressor if task=="regression" else MLPClassifier

    elif "XGBoost" in name and HAS_XGB:
        p = {"n_estimators":trial.suggest_int("n_estimators",50,500),
             "learning_rate":trial.suggest_float("learning_rate",0.01,0.3,log=True),
             "max_depth":trial.suggest_int("max_depth",2,10),
             "subsample":trial.suggest_float("subsample",0.5,1.0),
             "colsample_bytree":trial.suggest_float("colsample_bytree",0.5,1.0),
             "reg_alpha":trial.suggest_float("reg_alpha",1e-4,10,log=True),
             "random_state":42,"verbosity":0}
        cls = XGBRegressor if task=="regression" else XGBClassifier

    elif "LightGBM" in name and HAS_LGB:
        p = {"n_estimators":trial.suggest_int("n_estimators",50,500),
             "learning_rate":trial.suggest_float("learning_rate",0.01,0.3,log=True),
             "num_leaves":trial.suggest_int("num_leaves",15,127),
             "max_depth":trial.suggest_int("max_depth",3,12),
             "subsample":trial.suggest_float("subsample",0.5,1.0),
             "random_state":42,"verbose":-1}
        cls = LGBMRegressor if task=="regression" else LGBMClassifier

    elif name in ("Ridge","Ridge Regression"):
        p  = {"alpha":trial.suggest_float("alpha",1e-4,100,log=True)}
        cls = Ridge

    elif name == "Lasso":
        p  = {"alpha":trial.suggest_float("alpha",1e-4,10,log=True),"max_iter":5000}
        cls = Lasso

    else:
        p  = {"n_estimators":trial.suggest_int("n_estimators",50,300),"random_state":42}
        cls = RandomForestRegressor if task=="regression" else RandomForestClassifier

    return cls(**p)


# ══════════════════════════════════════════════════════════════════
# ── SECTION 4: PLOTS ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def plot_distributions(df):
    cols = df.select_dtypes("number").columns.tolist()[:12]
    if not cols: return None
    c = min(4, len(cols)); r = (len(cols)+c-1)//c
    fig = make_subplots(rows=r, cols=c, subplot_titles=cols)
    for i, col in enumerate(cols):
        rr,cc = divmod(i,c)
        fig.add_trace(go.Histogram(x=df[col], name=col, showlegend=False,
                      marker_color=BLUE, opacity=0.75), row=rr+1, col=cc+1)
    fig.update_layout(**LAYOUT, height=220*r, title="Feature Distributions")
    fig.update_xaxes(gridcolor=BORDER); fig.update_yaxes(gridcolor=BORDER)
    return fig


def plot_corr(df):
    num = df.select_dtypes("number")
    if num.shape[1] < 2: return None
    corr = num.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,RED],[0.5,SURFACE],[1,BLUE]], zmid=0,
        text=np.round(corr.values,2), texttemplate="%{text}", textfont_size=10
    ))
    return _fig(fig, "Correlation Matrix", max(400,60*len(corr)))


def plot_class_dist(y, title="Class Distribution"):
    u,c = np.unique(y, return_counts=True)
    fig = go.Figure(go.Bar(x=[str(v) for v in u], y=c,
                   marker_color=BLUE, text=c, textposition="outside"))
    return _fig(fig, title, 300)


def plot_boxes(df):
    cols = df.select_dtypes("number").columns[:8]
    fig  = go.Figure()
    clrs = px.colors.qualitative.Set2
    for i,col in enumerate(cols):
        fig.add_trace(go.Box(y=df[col], name=col,
                     marker_color=clrs[i%len(clrs)], boxmean=True))
    return _fig(fig, "Feature Spread", 400)


def plot_conf_matrix(cm, labels=None):
    arr = np.array(cm); n = arr.shape[0]
    lbl = labels or [str(i) for i in range(n)]
    fig = go.Figure(go.Heatmap(
        z=arr, x=[f"Pred:{l}" for l in lbl], y=[f"Act:{l}" for l in lbl],
        colorscale=[[0,SURFACE],[0.5,"#1f3a5f"],[1,BLUE]],
        text=[[str(arr[i][j]) for j in range(n)] for i in range(n)],
        texttemplate="%{text}", textfont=dict(size=18,family="Space Mono")
    ))
    return _fig(fig, "Confusion Matrix", 380)


def plot_roc(fpr, tpr, auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                  name=f"ROC (AUC={auc:.3f})",
                  line=dict(color=BLUE,width=2.5),
                  fill="tozeroy", fillcolor=BLUE+"22"))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1], mode="lines",
                  name="Random", line=dict(color=MUTED,dash="dash")))
    fig.update_xaxes(title="FPR"); fig.update_yaxes(title="TPR")
    return _fig(fig, "ROC Curve", 400)


def plot_pr(precision, recall, ap=None):
    lbl = f"PR Curve" + (f" (AP={ap:.3f})" if ap else "")
    fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines",
                  name=lbl, line=dict(color=GREEN,width=2.5),
                  fill="tozeroy", fillcolor=GREEN+"22"))
    fig.update_xaxes(title="Recall"); fig.update_yaxes(title="Precision")
    return _fig(fig, "Precision-Recall Curve", 400)


def plot_score_dist(y_true, p1, thr=0.5):
    pos = p1[y_true==1]; neg = p1[y_true==0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=neg, name="Class 0", opacity=0.7, marker_color=RED, nbinsx=40))
    fig.add_trace(go.Histogram(x=pos, name="Class 1", opacity=0.7, marker_color=GREEN, nbinsx=40))
    fig.add_vline(x=thr, line_dash="dash", line_color=YELLOW,
                  annotation_text=f"thr={thr}")
    fig.update_layout(barmode="overlay")
    return _fig(fig, "Score Distribution", 360)


def plot_radar(m, name):
    keys = ["accuracy","precision","recall","f1","roc_auc"]
    lbls = ["Accuracy","Precision","Recall","F1","AUC-ROC","Accuracy"]
    vals = [m.get(k,0) for k in keys] + [m.get(keys[0],0)]
    fig  = go.Figure(go.Scatterpolar(r=vals, theta=lbls, fill="toself",
                    line_color=BLUE, fillcolor=BLUE+"33", name=name))
    fig.update_layout(**LAYOUT, height=380,
        polar=dict(radialaxis=dict(visible=True,range=[0,1],
                   gridcolor=BORDER,color=MUTED),
                   angularaxis=dict(gridcolor=BORDER), bgcolor=SURFACE))
    return fig


def plot_multi_radar(results):
    clrs = [BLUE,GREEN,PURPLE,YELLOW,RED]
    keys = ["accuracy","precision","recall","f1"]
    lbls = ["Accuracy","Precision","Recall","F1","Accuracy"]
    fig  = go.Figure()
    for i,(name,m) in enumerate(results.items()):
        vals = [m.get(k,0) for k in keys] + [m.get(keys[0],0)]
        fig.add_trace(go.Scatterpolar(r=vals, theta=lbls, name=name,
                      line_color=clrs[i%len(clrs)],
                      fillcolor=clrs[i%len(clrs)]+"22", fill="toself"))
    fig.update_layout(**LAYOUT, height=420, title="Multi-Model Radar",
        polar=dict(radialaxis=dict(visible=True,range=[0,1],
                   gridcolor=BORDER,color=MUTED),
                   angularaxis=dict(gridcolor=BORDER), bgcolor=SURFACE))
    return fig


def plot_actual_vs_pred(yt, yp):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yt, y=yp, mode="markers",
                  marker=dict(color=BLUE,opacity=0.5,size=4), name="Samples"))
    mn,mx = min(min(yt),min(yp)), max(max(yt),max(yp))
    fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx], mode="lines",
                  line=dict(color=GREEN,dash="dash"), name="Perfect Fit"))
    fig.update_xaxes(title="Actual"); fig.update_yaxes(title="Predicted")
    return _fig(fig, "Actual vs Predicted", 400)


def plot_residuals(yt, yp):
    res = np.array(yt)-np.array(yp)
    fig = make_subplots(rows=1,cols=2,
          subplot_titles=["Residuals vs Predicted","Residual Distribution"])
    fig.add_trace(go.Scatter(x=list(yp),y=list(res), mode="markers",
                  marker=dict(color=BLUE,opacity=0.5,size=4)), row=1,col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=GREEN, row=1, col=1)
    fig.add_trace(go.Histogram(x=list(res), marker_color=PURPLE,
                  opacity=0.8, nbinsx=40), row=1,col=2)
    fig.update_layout(**LAYOUT, height=380, title="Residual Analysis", showlegend=False)
    fig.update_xaxes(gridcolor=BORDER); fig.update_yaxes(gridcolor=BORDER)
    return fig


def plot_feat_imp(df_imp, top_n=15):
    df = df_imp.head(top_n).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["feature"], orientation="h",
        marker=dict(color=df["importance"],
                    colorscale=[[0,SURFACE],[0.5,BLUE],[1,PURPLE]]),
        text=[f"{v:.3f}" for v in df["importance"]], textposition="outside"
    ))
    fig.update_xaxes(title="Importance")
    return _fig(fig, f"Feature Importance (Top {top_n})", max(350, 28*top_n))


def plot_model_bar(results, metric):
    names  = list(results.keys())
    values = [results[n].get(metric,0) for n in names]
    colors = [GREEN if v==max(values) else BLUE for v in values]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors,
                   text=[f"{v:.4f}" for v in values], textposition="outside"))
    fig.update_xaxes(tickangle=-25)
    return _fig(fig, f"Model Comparison — {metric.upper()}", 380)


def plot_opt_history(history):
    if not history: return None
    trials = [h[0] for h in history]
    scores = [h[1] for h in history]
    best   = [max(scores[:i+1]) for i in range(len(scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trials,y=scores, mode="markers",
                  marker=dict(color=BLUE,size=5,opacity=0.6), name="Trial"))
    fig.add_trace(go.Scatter(x=trials,y=best, mode="lines",
                  line=dict(color=GREEN,width=2.5), name="Best so far"))
    fig.update_xaxes(title="Trial"); fig.update_yaxes(title="Score")
    return _fig(fig, "Optimization History", 380)


def plot_param_imp(imp_dict):
    if not imp_dict: return None
    names  = list(imp_dict.keys()); values = list(imp_dict.values())
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=PURPLE,
                   text=[f"{v:.3f}" for v in values], textposition="outside"))
    return _fig(fig, "Hyperparameter Importance", 320)


def plot_parallel(t_df, score_col="value"):
    if t_df is None or len(t_df)<3: return None
    pcols = [c for c in t_df.columns if c not in (score_col,"number","state")][:8]
    dims  = [dict(label=score_col.upper(), values=t_df[score_col],
                  range=[t_df[score_col].min(), t_df[score_col].max()])]
    for col in pcols:
        s = t_df[col].copy()
        if s.dtype==object:
            vm={v:i for i,v in enumerate(s.unique())}; s=s.map(vm)
        dims.append(dict(label=col, values=s))
    fig = go.Figure(go.Parcoords(
        line=dict(color=t_df[score_col],
                  colorscale=[[0,RED],[0.5,YELLOW],[1,GREEN]], showscale=True),
        dimensions=dims))
    fig.update_layout(**LAYOUT, height=420, title="Parallel Coordinates — All Trials")
    return fig


# ══════════════════════════════════════════════════════════════════
# ── SECTION 5: LOG EXPERIMENT ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def log_exp(name, metrics, task, train_time, tag=""):
    m = metrics
    entry = {
        "timestamp" : datetime.datetime.now().strftime("%H:%M:%S"),
        "model"     : name + (f" [{tag}]" if tag else ""),
        "task"      : task,
        "accuracy"  : round(m.get("accuracy",0),4),
        "f1"        : round(m.get("f1",0),4),
        "roc_auc"   : round(m.get("roc_auc",0),4),
        "r2"        : round(m.get("r2",0),4),
        "rmse"      : round(m.get("rmse",0),4),
        "cv_mean"   : round(m.get("cv_mean",0),4),
        "train_time": train_time,
    }
    st.session_state.setdefault("experiments",[]).append(entry)


# ══════════════════════════════════════════════════════════════════
# ── SECTION 6: PAGE — HOME ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def page_home():
    st.markdown('<div class="hero-title">⚡ VertexFlow</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">End-to-end ML Platform · Data · Train · Evaluate · Optimize</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    features = [
        ("📊","Data Studio","Synthetic generator · CSV/Excel upload · Quality profiling · EDA suite"),
        ("🔬","Model Lab","15+ algorithms · Multi-model training · Cross-validation · Feature importance"),
        ("📈","Evaluation Suite","Full metric suite · ROC/PR curves · Confusion matrix · Threshold slider · Residual analysis"),
        ("🧠","Model Optimizer ⭐","Bayesian search (Optuna) · Optimization history · Param importance · Parallel coords"),
        ("📋","Experiment Tracker","Auto-log every run · Side-by-side comparison · CSV export"),
        ("🔄","Full Pipeline","Seamless data → train → evaluate → optimize flow with shared session state"),
    ]
    cols = st.columns(3)
    for i,(icon,title,desc) in enumerate(features):
        border = "#1f6feb" if "⭐" in title else BORDER
        with cols[i%3]:
            st.markdown(f"""
            <div class="info-card" style="border:1px solid {border}; min-height:170px; margin-bottom:14px;">
                <div style="font-size:24px;margin-bottom:6px;">{icon}</div>
                <div style="font-weight:700;color:{TEXT};margin-bottom:6px;">{title}</div>
                <div style="color:{MUTED};font-size:13px;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">CURRENT SESSION</div>', unsafe_allow_html=True)
    df      = st.session_state.get("dataset")
    trained = st.session_state.get("trained_models",{})
    exps    = st.session_state.get("experiments",[])
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Dataset",  f"{len(df)} rows" if df is not None else "None")
    c2.metric("Task",     st.session_state.get("task_type","—").title())
    c3.metric("Models",   len(trained))
    c4.metric("Runs",     len(exps))

    if trained:
        st.markdown('<div class="section-header">LEADERBOARD</div>', unsafe_allow_html=True)
        task = st.session_state.get("task_type","binary")
        rows = []
        for n,info in trained.items():
            m = info["metrics"]
            if task=="regression":
                rows.append({"Model":n,"R²":round(m.get("r2",0),4),
                             "RMSE":round(m.get("rmse",0),4),"MAE":round(m.get("mae",0),4)})
            else:
                rows.append({"Model":n,"Accuracy":f"{m.get('accuracy',0)*100:.2f}%",
                             "F1":round(m.get("f1",0),4),"AUC":round(m.get("roc_auc",0),4)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# ── SECTION 7: PAGE — DATA STUDIO ────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def page_data():
    st.markdown('<div class="hero-title" style="font-size:2rem;">📊 Data Studio</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Prepare, profile, and explore your dataset</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    tab_syn, tab_bi, tab_up = st.tabs(
        ["🔧 Synthetic Generator","📦 Built-in Datasets","📁 Upload CSV/Excel"])

    with tab_syn:
        st.markdown('<div class="section-header">SYNTHETIC DATA GENERATOR</div>',
                    unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        tmpl  = c1.selectbox("Template", list(SYNTHETIC_TEMPLATES.keys()))
        nsamp = c2.slider("Samples",100,5000,500,step=50)
        nfeat = c3.slider("Features",4,30,8)
        c4,c5 = st.columns(2)
        noise = c4.slider("Noise",0.0,0.5,0.1,step=0.05)
        seed  = c5.number_input("Seed",value=42,step=1)
        if st.button("⚡ Generate Dataset", type="primary"):
            df,tc,task = gen_synthetic(SYNTHETIC_TEMPLATES[tmpl],nsamp,nfeat,noise,int(seed))
            _save_ds(df,tc,task)
            st.success(f"✓ Generated {len(df)} rows × {len(df.columns)} cols · Task: **{task}**")

    with tab_bi:
        st.markdown('<div class="section-header">BUILT-IN SKLEARN DATASETS</div>',
                    unsafe_allow_html=True)
        ds = st.selectbox("Dataset", list(BUILTIN_DATASETS.keys()))
        st.info(DS_DESC.get(ds,""))
        if st.button("📦 Load Dataset", type="primary"):
            with st.spinner("Loading…"):
                df,tc,task = load_builtin(ds)
            _save_ds(df,tc,task)
            st.success(f"✓ {ds} — {len(df)} rows · target=`{tc}` · task=`{task}`")

    with tab_up:
        st.markdown('<div class="section-header">UPLOAD YOUR DATA</div>',
                    unsafe_allow_html=True)
        f = st.file_uploader("CSV or Excel", type=["csv","xlsx","xls"])
        if f:
            try:
                df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
                tc   = st.selectbox("Target column", df.columns.tolist(),
                                    index=len(df.columns)-1)
                task = st.selectbox("Task type",["binary","multiclass","regression"],
                                    index=["binary","multiclass","regression"].index(auto_task(df,tc)))
                if st.button("✓ Load", type="primary"):
                    _save_ds(df,tc,task)
                    st.success(f"✓ Loaded `{f.name}` — {len(df)} rows")
            except Exception as e:
                st.error(f"Error: {e}")

    # ── EDA ──────────────────────────────────────────────────────────
    df = st.session_state.get("dataset")
    if df is None:
        st.warning("⚠ Load a dataset above to proceed.")
        return

    tc   = st.session_state["target_col"]
    st.markdown("---")

    # Quality
    st.markdown('<div class="section-header">DATA QUALITY</div>', unsafe_allow_html=True)
    score, col_df = data_quality(df)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Quality Score",f"{score}/100")
    c2.metric("Rows",len(df)); c3.metric("Columns",len(df.columns))
    c4.metric("Missing",int(df.isnull().sum().sum()))
    c5.metric("Duplicates",int(df.duplicated().sum()))
    with st.expander("📋 Column profile"):
        st.dataframe(col_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">PREVIEW</div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<div class="section-header">EXPLORATORY ANALYSIS</div>',
                unsafe_allow_html=True)
    t1,t2,t3,t4,t5 = st.tabs(
        ["📊 Distributions","🔥 Correlations","📦 Box Plots","🎯 Target","🔍 Scatter"])

    with t1:
        fig = plot_distributions(df)
        if fig: st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig = plot_corr(df)
        if fig: st.plotly_chart(fig, use_container_width=True)
        if tc in df.select_dtypes("number").columns:
            ct = df.select_dtypes("number").corr()[tc].drop(tc).abs().sort_values(ascending=False)
            st.markdown("**Top correlations with target:**")
            st.dataframe(ct.reset_index().rename(columns={"index":"Feature",tc:"|Corr|"}).head(10),
                         use_container_width=True, hide_index=True)

    with t3:
        st.plotly_chart(plot_boxes(df), use_container_width=True)

    with t4:
        st.plotly_chart(plot_class_dist(df[tc], f"Target: {tc}"), use_container_width=True)
        vc = df[tc].value_counts()
        st.dataframe(vc.reset_index().rename(columns={"index":"Value",tc:"Count"}),
                     use_container_width=True, hide_index=True)

    with t5:
        nums = df.select_dtypes("number").columns.tolist()
        if len(nums)>=2:
            c1,c2 = st.columns(2)
            xc = c1.selectbox("X",nums,0); yc = c2.selectbox("Y",nums,min(1,len(nums)-1))
            clrc = st.selectbox("Color by",[None,tc]+nums[:5])
            kwargs = dict(x=xc,y=yc,opacity=0.7)
            if clrc: kwargs["color"]=clrc
            fig = px.scatter(df,**kwargs)
            _fig(fig, f"{xc} vs {yc}", 380)
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("📐 Descriptive Statistics"):
        st.dataframe(df.describe().T, use_container_width=True)

    st.markdown("---")
    st.download_button("⬇ Download Dataset", df.to_csv(index=False).encode(),
                       "dataset.csv","text/csv")


def _save_ds(df, tc, task):
    st.session_state["dataset"]    = df
    st.session_state["target_col"] = tc
    st.session_state["task_type"]  = task
    st.session_state.pop("trained_models",None)


# ══════════════════════════════════════════════════════════════════
# ── SECTION 8: PAGE — MODEL LAB ───────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def page_model_lab():
    st.markdown('<div class="hero-title" style="font-size:2rem;">🔬 Model Lab</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Train and cross-validate multiple models simultaneously</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    df = st.session_state.get("dataset")
    if df is None:
        st.warning("⚠ No dataset. Go to 📊 Data Studio first."); return

    tc      = st.session_state["target_col"]
    task    = st.session_state["task_type"]
    catalog = REG_MODELS if task=="regression" else CLF_MODELS

    st.markdown('<div class="section-header">CONFIGURATION</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.markdown(f"**Task:** `{task}` · **Target:** `{tc}`")
    scale  = c2.checkbox("StandardScale features", True)
    cv_folds = c3.slider("CV folds",3,10,5)

    families = sorted(set(v["family"] for v in catalog.values()))
    sel_fam  = st.multiselect("Filter by family",families,default=families)
    filtered = {k:v for k,v in catalog.items() if v["family"] in sel_fam}

    st.markdown('<div class="section-header">SELECT MODELS</div>', unsafe_allow_html=True)
    all_names    = list(filtered.keys())
    sel_models   = st.multiselect("Models to train", all_names,
                                  default=all_names[:min(4,len(all_names))])

    if not sel_models:
        st.info("Select at least one model."); return

    if st.button("🚀 Train Selected Models", type="primary"):
        with st.spinner("Preprocessing…"):
            try:
                X_tr,X_te,y_tr,y_te,feats,sc = preprocess(df,tc,scale)
                for k,v in [("X_train",X_tr),("X_test",X_te),("y_train",y_tr),
                             ("y_test",y_te),("feature_names",feats),("scaler",sc)]:
                    st.session_state[k] = v
            except Exception as e:
                st.error(f"Preprocessing failed: {e}"); return

        st.info(f"Train: {len(X_tr)} · Test: {len(X_te)} · Features: {len(feats)}")
        trained  = {}
        prog     = st.progress(0)
        status   = st.empty()

        for idx,name in enumerate(sel_models):
            status.markdown(f"⚙ Training **{name}**…")
            t0  = time.time()
            cfg = catalog[name]
            try:
                model = cfg["cls"](**cfg["params"])
                model.fit(X_tr, y_tr)
                elapsed = round(time.time()-t0,2)
                y_pred  = model.predict(X_te)
                y_proba = None
                if task!="regression" and hasattr(model,"predict_proba"):
                    y_proba = model.predict_proba(X_te)

                metrics = (reg_metrics(y_te,y_pred) if task=="regression"
                           else clf_metrics(y_te,y_pred,y_proba,task))

                cv = (StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=42)
                      if task!="regression"
                      else KFold(n_splits=cv_folds,shuffle=True,random_state=42))
                sc_name = "accuracy" if task!="regression" else "r2"
                cv_sc = cross_val_score(cfg["cls"](**cfg["params"]),
                                        X_tr,y_tr,cv=cv,scoring=sc_name,n_jobs=-1)
                metrics["cv_mean"]   = float(cv_sc.mean())
                metrics["cv_std"]    = float(cv_sc.std())
                metrics["cv_scores"] = cv_sc.tolist()

                imp = feat_importance(model,X_te,y_te,feats,task)
                trained[name] = {"model":model,"metrics":metrics,"importance":imp,
                                 "train_time":elapsed,
                                 "y_pred":y_pred.tolist(),
                                 "y_proba":y_proba.tolist() if y_proba is not None else None}
                log_exp(name, metrics, task, elapsed)
            except Exception as e:
                st.warning(f"⚠ {name} failed: {e}")
            prog.progress((idx+1)/len(sel_models))

        status.empty()
        st.session_state["trained_models"] = trained
        st.success(f"✅ Trained {len(trained)} model(s)!")

    # ── Results ───────────────────────────────────────────────────────
    trained = st.session_state.get("trained_models",{})
    if not trained: return

    st.markdown("---")
    st.markdown('<div class="section-header">LEADERBOARD</div>', unsafe_allow_html=True)

    rows = []
    for n,info in trained.items():
        m = info["metrics"]
        if task=="regression":
            rows.append({"Model":n,"R²":round(m.get("r2",0),4),
                         "RMSE":round(m.get("rmse",0),4),"MAE":round(m.get("mae",0),4),
                         "MAPE%":round(m.get("mape",0),2),
                         "CV R² (mean±std)":f"{m.get('cv_mean',0):.3f}±{m.get('cv_std',0):.3f}",
                         "Time(s)":info["train_time"]})
        else:
            rows.append({"Model":n,"Accuracy":f"{m.get('accuracy',0)*100:.2f}%",
                         "Precision":round(m.get("precision",0),4),
                         "Recall":round(m.get("recall",0),4),
                         "F1":round(m.get("f1",0),4),
                         "AUC-ROC":round(m.get("roc_auc",0),4),
                         "Log Loss":round(m.get("log_loss",9999),4),
                         "CV (mean±std)":f"{m.get('cv_mean',0)*100:.1f}%±{m.get('cv_std',0)*100:.1f}%",
                         "Time(s)":info["train_time"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if len(trained)>1:
        st.markdown('<div class="section-header">COMPARISON</div>', unsafe_allow_html=True)
        mk = st.selectbox("Compare by",
             (["r2","rmse","mae"] if task=="regression"
              else ["accuracy","f1","precision","recall","roc_auc"]))
        st.plotly_chart(plot_model_bar({n:info["metrics"] for n,info in trained.items()},mk),
                        use_container_width=True)
        if task!="regression":
            st.plotly_chart(plot_multi_radar({n:info["metrics"] for n,info in trained.items()}),
                            use_container_width=True)

    st.markdown('<div class="section-header">PER-MODEL DETAILS</div>',
                unsafe_allow_html=True)
    for name,info in trained.items():
        with st.expander(f"🔍 {name}"):
            m  = info["metrics"]
            cc = st.columns(4)
            if task=="regression":
                cc[0].metric("R²",round(m.get("r2",0),4))
                cc[1].metric("RMSE",round(m.get("rmse",0),4))
                cc[2].metric("MAE",round(m.get("mae",0),4))
                cc[3].metric("MAPE",f"{round(m.get('mape',0),2)}%")
            else:
                cc[0].metric("Accuracy",f"{m.get('accuracy',0)*100:.2f}%")
                cc[1].metric("F1",round(m.get("f1",0),4))
                cc[2].metric("AUC-ROC",round(m.get("roc_auc",0),4))
                cc[3].metric("Log Loss",round(m.get("log_loss",9999),4))
            cvs = m.get("cv_scores",[])
            if cvs:
                st.markdown(f"**CV:** {' | '.join([f'{s:.3f}' for s in cvs])} → "
                            f"mean={m.get('cv_mean',0):.4f} std={m.get('cv_std',0):.4f}")
            imp = info.get("importance")
            if imp is not None and not imp.empty:
                st.plotly_chart(plot_feat_imp(imp), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# ── SECTION 9: PAGE — EVALUATION ──────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def page_eval():
    st.markdown('<div class="hero-title" style="font-size:2rem;">📈 Evaluation Suite</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Deep-dive metrics, curves, and model comparison</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    trained = st.session_state.get("trained_models",{})
    if not trained:
        st.warning("⚠ No trained models. Go to 🔬 Model Lab first."); return

    task   = st.session_state.get("task_type","binary")
    y_test = np.array(st.session_state.get("y_test",[]))
    sel    = st.selectbox("Select model", list(trained.keys()))
    info   = trained[sel]; m = info["metrics"]

    # Metrics row
    st.markdown('<div class="section-header">KEY METRICS</div>', unsafe_allow_html=True)
    if task=="regression":
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("R²",round(m.get("r2",0),4))
        c2.metric("RMSE",round(m.get("rmse",0),4))
        c3.metric("MAE",round(m.get("mae",0),4))
        c4.metric("MAPE",f"{round(m.get('mape',0),2)}%")
        c5.metric("Explained Var",round(m.get("explained_var",0),4))
    else:
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Accuracy",f"{m.get('accuracy',0)*100:.2f}%")
        c2.metric("Precision",round(m.get("precision",0),4))
        c3.metric("Recall",round(m.get("recall",0),4))
        c4.metric("F1",round(m.get("f1",0),4))
        c5.metric("AUC-ROC",round(m.get("roc_auc",0),4))
        c6.metric("Log Loss",round(m.get("log_loss",9999),4))
        cc = st.columns(4)
        cc[0].metric("F1 Macro",round(m.get("f1_macro",0),4))
        cc[1].metric("F1 Micro",round(m.get("f1_micro",0),4))
        cc[2].metric("Avg Precision",round(m.get("avg_precision",0),4))
        cc[3].metric("CV Score",f"{m.get('cv_mean',0)*100:.2f}%±{m.get('cv_std',0)*100:.2f}%")

    st.markdown("---")

    if task=="regression":
        yp = np.array(m.get("y_pred",[])); yt = np.array(m.get("y_true",y_test))
        t1,t2 = st.tabs(["🎯 Actual vs Predicted","📉 Residuals"])
        with t1: st.plotly_chart(plot_actual_vs_pred(yt.tolist(),yp.tolist()),use_container_width=True)
        with t2: st.plotly_chart(plot_residuals(yt.tolist(),yp.tolist()),use_container_width=True)
    else:
        y_pred  = np.array(info.get("y_pred",[]))
        y_proba = info.get("y_proba")
        t1,t2,t3,t4,t5 = st.tabs(
            ["🟦 Confusion Matrix","📈 ROC","🎯 PR Curve","📊 Score Dist.","🕸 Radar"])

        with t1:
            n = len(np.unique(y_test))
            st.plotly_chart(plot_conf_matrix(m.get("conf_matrix",[]),
                            [str(i) for i in range(n)]),use_container_width=True)
            if task=="binary" and len(m.get("conf_matrix",[]))==2:
                cm = m["conf_matrix"]
                tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("True Positive",tp); c2.metric("True Negative",tn)
                c3.metric("False Positive",fp); c4.metric("False Negative",fn)

        with t2:
            roc = m.get("roc_curve")
            if roc:
                st.plotly_chart(plot_roc(roc["fpr"],roc["tpr"],m.get("roc_auc",0)),
                                use_container_width=True)
                auc = m.get("roc_auc",0)
                rating = "Excellent" if auc>0.9 else "Good" if auc>0.8 else "Fair" if auc>0.7 else "Poor"
                st.info(f"AUC = {auc:.4f} — {rating}")
            else: st.info("ROC requires binary classification.")

        with t3:
            pr = m.get("pr_curve")
            if pr: st.plotly_chart(plot_pr(pr["precision"],pr["recall"],m.get("avg_precision")),use_container_width=True)
            else:  st.info("PR curve requires binary classification.")

        with t4:
            if y_proba is not None and task=="binary":
                pa = np.array(y_proba)
                p1 = pa[:,1] if pa.ndim==2 else pa
                thr = st.slider("Threshold",0.01,0.99,0.5,step=0.01)
                st.plotly_chart(plot_score_dist(y_test,p1,thr),use_container_width=True)
                y_thr = (p1>=thr).astype(int)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Acc@thr",f"{accuracy_score(y_test,y_thr)*100:.2f}%")
                c2.metric("Prec@thr",f"{precision_score(y_test,y_thr,zero_division=0):.4f}")
                c3.metric("Rec@thr", f"{recall_score(y_test,y_thr,zero_division=0):.4f}")
                c4.metric("F1@thr",  f"{f1_score(y_test,y_thr,zero_division=0):.4f}")
            else: st.info("Score distribution requires binary classification with predict_proba.")

        with t5:
            st.plotly_chart(plot_radar(m,sel),use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.markdown('<div class="section-header">FEATURE IMPORTANCE</div>',unsafe_allow_html=True)
    imp = info.get("importance")
    if imp is not None and not imp.empty:
        top = st.slider("Top N features",5,min(30,len(imp)),min(15,len(imp)))
        st.plotly_chart(plot_feat_imp(imp,top),use_container_width=True)
        with st.expander("Full table"): st.dataframe(imp,use_container_width=True,hide_index=True)

    # Multi-model comparison
    if len(trained)>1:
        st.markdown("---")
        st.markdown('<div class="section-header">MULTI-MODEL COMPARISON</div>',
                    unsafe_allow_html=True)
        mlist = (["r2","rmse","mae"] if task=="regression"
                 else ["accuracy","f1","precision","recall","roc_auc"])
        for metric in mlist:
            st.plotly_chart(
                plot_model_bar({n:trained[n]["metrics"] for n in trained},metric),
                use_container_width=True)
        if task!="regression":
            st.plotly_chart(plot_multi_radar({n:trained[n]["metrics"] for n in trained}),
                            use_container_width=True)

    # Classification report
    if task!="regression" and "report" in m:
        st.markdown("---")
        with st.expander("📋 Full Classification Report"):
            rows = []
            for label,vals in m["report"].items():
                if isinstance(vals,dict):
                    rows.append({"Class":label,
                                 "Precision":round(vals.get("precision",0),4),
                                 "Recall":round(vals.get("recall",0),4),
                                 "F1":round(vals.get("f1-score",0),4),
                                 "Support":int(vals.get("support",0))})
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)


# ══════════════════════════════════════════════════════════════════
# ── SECTION 10: PAGE — OPTIMIZER (USP) ────────────────────────────
# ══════════════════════════════════════════════════════════════════

def page_optimizer():
    st.markdown("""
    <div style="padding:10px 0 4px 0;">
        <div class="hero-title" style="font-size:2rem;">🧠 Model Optimizer</div>
        <span class="badge badge-purple" style="margin-top:6px;display:inline-block;">⭐ USP</span>
        <div class="hero-sub" style="margin-top:8px;">
            Bayesian hyperparameter search with Optuna — find the best config automatically
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    X_tr = st.session_state.get("X_train")
    X_te = st.session_state.get("X_test")
    y_tr = st.session_state.get("y_train")
    y_te = st.session_state.get("y_test")
    feats= st.session_state.get("feature_names",[])

    if X_tr is None:
        st.warning("⚠ Complete Data Studio → Model Lab first."); return

    task    = st.session_state.get("task_type","binary")
    catalog = REG_MODELS if task=="regression" else CLF_MODELS

    if not HAS_OPTUNA:
        st.error("Optuna not installed. Run: `pip install optuna`")
        _grid_search_fallback(X_tr,X_te,y_tr,y_te,task,catalog,feats)
        return

    st.markdown('<div class="section-header">OPTIMIZATION CONFIGURATION</div>',
                unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    model_name = c1.selectbox("Model to optimize", list(catalog.keys()))
    n_trials   = c2.slider("Trials",10,200,50,step=10)
    cv_folds   = c3.slider("CV folds",3,10,5)

    c4,c5 = st.columns(2)
    direction  = c4.selectbox("Direction",["maximize","minimize"],
                               index=0 if task!="regression" else 1)
    scoring    = c5.selectbox("Objective metric",
        (["accuracy","f1_weighted","roc_auc","precision_weighted","recall_weighted"]
         if task!="regression"
         else ["r2","neg_mean_squared_error","neg_mean_absolute_error"]))

    info = catalog[model_name]
    st.markdown(f"""
    <div class="info-card accent-card">
        <b>{model_name}</b> — Family: <b>{info['family']}</b> · Task: <b>{task}</b> ·
        Trials: <b>{n_trials}</b> · CV: <b>{cv_folds}</b>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    if st.button("🚀 Start Optimization", type="primary"):
        cv = (StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=42)
              if task!="regression"
              else KFold(n_splits=cv_folds,shuffle=True,random_state=42))

        history = []
        prog    = st.progress(0)
        status  = st.empty()

        def objective(trial):
            model = optuna_space(trial, model_name, task)
            scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()

        def callback(study, trial):
            n = len(study.trials)
            prog.progress(min(n/n_trials,1.0))
            history.append((n, trial.value))
            status.markdown(f"⚙ Trial **{n}/{n_trials}** — "
                            f"current: `{trial.value:.4f}` best: `{study.best_value:.4f}`")

        t0 = time.time()
        study = optuna.create_study(direction=direction,
                                    sampler=optuna.samplers.TPESampler(seed=42))
        with st.spinner(f"Running {n_trials} Bayesian trials…"):
            study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        elapsed = round(time.time()-t0,1)
        status.empty(); prog.progress(1.0)
        st.success(f"✅ Done in **{elapsed}s** — Best {scoring}: **{study.best_value:.4f}**")

        # Train best model on full train set
        class _MT:
            def __init__(self,p): self._p=p
            def suggest_int(self,n,*a,**k): return self._p.get(n,a[0])
            def suggest_float(self,n,*a,**k): return self._p.get(n,a[0])
            def suggest_categorical(self,n,c,**k): return self._p.get(n,c[0])

        best_model = optuna_space(_MT(study.best_params), model_name, task)
        best_model.fit(X_tr, y_tr)
        y_pred  = best_model.predict(X_te)
        y_proba = None
        if task!="regression" and hasattr(best_model,"predict_proba"):
            y_proba = best_model.predict_proba(X_te)
        best_m  = (reg_metrics(y_te,y_pred) if task=="regression"
                   else clf_metrics(y_te,y_pred,y_proba,task))
        imp_df  = feat_importance(best_model,X_te,y_te,feats,task)

        opt = {"model_name":model_name,"best_params":study.best_params,
               "best_score":study.best_value,"history":history,
               "study":study,"best_metrics":best_m,
               "importance":imp_df,"elapsed":elapsed,"scoring":scoring}
        st.session_state["opt_result"] = opt
        log_exp(model_name, best_m, task, elapsed, tag="OPTIMIZED")

    # ── Show results ──────────────────────────────────────────────────
    opt = st.session_state.get("opt_result")
    if opt is None:
        st.info("Configure above and click Start Optimization."); return

    st.markdown("---")
    st.markdown('<div class="section-header">RESULTS</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Best Score",round(opt["best_score"],4))
    c2.metric("Metric",opt["scoring"])
    c3.metric("Model",opt["model_name"])
    c4.metric("Time",f"{opt['elapsed']}s")

    st.markdown('<div class="section-header">BEST HYPERPARAMETERS</div>',
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Parameter":k,"Value":v}
                                for k,v in opt["best_params"].items()]),
                 use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">TEST SET PERFORMANCE</div>',
                unsafe_allow_html=True)
    bm = opt["best_metrics"]
    if task=="regression":
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("R²",round(bm.get("r2",0),4))
        c2.metric("RMSE",round(bm.get("rmse",0),4))
        c3.metric("MAE",round(bm.get("mae",0),4))
        c4.metric("MAPE",f"{round(bm.get('mape',0),2)}%")
    else:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Accuracy",f"{bm.get('accuracy',0)*100:.2f}%")
        c2.metric("Precision",round(bm.get("precision",0),4))
        c3.metric("Recall",round(bm.get("recall",0),4))
        c4.metric("F1",round(bm.get("f1",0),4))
        c5.metric("AUC-ROC",round(bm.get("roc_auc",0),4))

    # Baseline vs Optimized
    trained = st.session_state.get("trained_models",{})
    if opt["model_name"] in trained:
        st.markdown('<div class="section-header">BASELINE vs OPTIMIZED</div>',
                    unsafe_allow_html=True)
        base_m = trained[opt["model_name"]]["metrics"]
        ckeys  = (["r2","rmse","mae"] if task=="regression"
                  else ["accuracy","f1","precision","recall","roc_auc"])
        rows   = []
        for k in ckeys:
            bv,ov = base_m.get(k,0), bm.get(k,0)
            d     = ov-bv
            rows.append({"Metric":k.upper(),"Baseline":round(bv,4),"Optimized":round(ov,4),
                         "Δ":f"+{round(d,4)}" if d>=0 else str(round(d,4)),
                         "✓":" ✅" if (d>0 and k!="rmse") or (d<0 and k=="rmse") else "❌"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    # Viz tabs
    st.markdown("---")
    t1,t2,t3,t4 = st.tabs(
        ["📈 Opt. History","🎛 Param Importance","🔀 Parallel Coords","📊 Feature Importance"])

    with t1:
        fig = plot_opt_history(opt["history"])
        if fig: st.plotly_chart(fig,use_container_width=True)
        study = opt["study"]
        td    = []
        for t in study.trials:
            row = {"trial":t.number,"value":round(t.value,4) if t.value else None}
            row.update(t.params); td.append(row)
        if td:
            with st.expander("All trials"):
                st.dataframe(pd.DataFrame(td),use_container_width=True,hide_index=True)

    with t2:
        try:
            imp = optuna.importance.get_param_importances(opt["study"])
            if imp:
                st.plotly_chart(plot_param_imp(imp),use_container_width=True)
                st.dataframe(pd.DataFrame([{"Hyperparameter":k,"Importance":round(v,4)}
                             for k,v in sorted(imp.items(),key=lambda x:-x[1])]),
                             use_container_width=True,hide_index=True)
                st.info("💡 Higher = more impact. Focus future tuning on top parameters.")
        except Exception as e:
            st.info(f"Need ≥2 completed trials. ({e})")

    with t3:
        study = opt["study"]
        tdf   = study.trials_dataframe(attrs=("number","value","params","state"))
        tdf   = tdf[tdf["state"]=="COMPLETE"].drop(columns=["state"])
        fig   = plot_parallel(tdf,"value")
        if fig: st.plotly_chart(fig,use_container_width=True)
        else:   st.info("Need ≥3 completed trials.")

    with t4:
        imp = opt.get("importance")
        if imp is not None and not imp.empty:
            st.plotly_chart(plot_feat_imp(imp),use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">EXPORT BEST PARAMS</div>',
                unsafe_allow_html=True)
    st.code(f"""# Best hyperparameters for {opt['model_name']}
best_params = {opt['best_params']}

# Re-train on full dataset
model = YourModel(**best_params)
model.fit(X_train, y_train)

# Save
import pickle
with open("best_model.pkl","wb") as f:
    pickle.dump(model, f)
""", language="python")


def _grid_search_fallback(X_tr,X_te,y_tr,y_te,task,catalog,feats):
    st.markdown('<div class="section-header">GRID SEARCH FALLBACK</div>',
                unsafe_allow_html=True)
    mname = st.selectbox("Model",list(catalog.keys()))
    grids = {
        "Random Forest"          : {"n_estimators":[50,100,200],"max_depth":[None,5,10]},
        "Random Forest (Reg.)"   : {"n_estimators":[50,100,200],"max_depth":[None,5,10]},
        "Gradient Boosting"      : {"n_estimators":[50,100],"learning_rate":[0.05,0.1,0.2]},
        "Logistic Regression"    : {"C":[0.01,0.1,1,10]},
        "Ridge"                  : {"alpha":[0.01,0.1,1,10,100]},
    }
    grid = grids.get(mname,{"n_estimators":[50,100]})
    st.json(grid)
    if st.button("🔎 Run Grid Search"):
        cfg = catalog[mname]
        gs  = GridSearchCV(cfg["cls"](**cfg["params"]),grid,
                           cv=5,scoring="accuracy" if task!="regression" else "r2",n_jobs=-1)
        with st.spinner("Running Grid Search…"):
            gs.fit(X_tr,y_tr)
        st.success(f"Best score: {gs.best_score_:.4f}")
        st.json(gs.best_params_)


# ══════════════════════════════════════════════════════════════════
# ── SECTION 11: PAGE — EXPERIMENTS ────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def page_experiments():
    st.markdown('<div class="hero-title" style="font-size:2rem;">📋 Experiment Tracker</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Compare all runs · track progress · export results</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    exps = st.session_state.get("experiments",[])
    if not exps:
        st.info("No experiments yet. Train models in 🔬 Model Lab or 🧠 Optimizer."); return

    df   = pd.DataFrame(exps)
    task = st.session_state.get("task_type","binary")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Runs",len(df))
    if task!="regression":
        c2.metric("Best Accuracy",f"{df['accuracy'].max()*100:.2f}%")
        c3.metric("Best F1",round(df["f1"].max(),4))
        c4.metric("Best AUC",round(df["roc_auc"].max(),4))
    else:
        c2.metric("Best R²",round(df["r2"].max(),4))
        c3.metric("Best RMSE",round(df["rmse"].min(),4))

    st.markdown("---")
    st.markdown('<div class="section-header">ALL EXPERIMENTS</div>',unsafe_allow_html=True)

    dcols = ["timestamp","model","task"]
    dcols += (["accuracy","f1","roc_auc","cv_mean","train_time"]
              if task!="regression" else ["r2","rmse","cv_mean","train_time"])
    dcols = [c for c in dcols if c in df.columns]
    st.dataframe(df[dcols],use_container_width=True,hide_index=True)

    st.markdown('<div class="section-header">PROGRESS OVER RUNS</div>',unsafe_allow_html=True)
    mopt = ["accuracy","f1","roc_auc"] if task!="regression" else ["r2","rmse"]
    mopt = [m for m in mopt if m in df.columns]
    sel  = st.selectbox("Metric",mopt)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1,len(df)+1)), y=df[sel].tolist(),
        mode="lines+markers+text",
        text=df["model"].str[:12].tolist(), textposition="top center",
        textfont=dict(size=8), marker=dict(size=7,color=BLUE),
        line=dict(color=BLUE,width=2), name=sel
    ))
    fig.update_xaxes(title="Run #",dtick=1); fig.update_yaxes(title=sel.upper())
    fig.update_layout(**LAYOUT,**GRID,height=360,title=f"{sel.upper()} over runs")
    st.plotly_chart(fig,use_container_width=True)

    fig2 = go.Figure(go.Bar(x=df["model"],y=df[sel],marker_color=BLUE,
                    text=[round(v,4) for v in df[sel]],textposition="outside"))
    fig2.update_layout(**LAYOUT,**GRID,height=360,title="All Models Compared")
    fig2.update_xaxes(tickangle=-30)
    st.plotly_chart(fig2,use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">BEST MODEL</div>',unsafe_allow_html=True)
    best = df.loc[df["f1"].idxmax() if task!="regression" else df["r2"].idxmax()]
    st.markdown(f"""
    <div class="info-card success-card">
        <div style="font-size:18px;font-weight:700;color:{GREEN};margin-bottom:6px;">
            🏆 {best.get('model','N/A')}
        </div>
        <div style="color:{MUTED};font-size:13px;">
            Logged at {best.get('timestamp','—')} · Task: {best.get('task','—')} ·
            CV: {round(best.get('cv_mean',0),4)} · Time: {best.get('train_time','—')}s
        </div>
    </div>""",unsafe_allow_html=True)

    st.markdown("---")
    st.download_button("⬇ Download Log (CSV)",
                       df.to_csv(index=False).encode(),"experiments.csv","text/csv")
    if st.button("🗑 Clear All"):
        st.session_state["experiments"] = []; st.rerun()


# ══════════════════════════════════════════════════════════════════
# ── SECTION 12: SIDEBAR + ROUTING ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:12px 0 24px 0;'>
        <div style='font-family:"Space Mono",monospace;font-size:20px;font-weight:700;
                    background:linear-gradient(135deg,#58a6ff,#a371f7);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            ⚡ VertexFlow
        </div>
        <div style='color:#8b949e;font-size:11px;margin-top:4px;'>ML Platform v1.0</div>
    </div>""",unsafe_allow_html=True)

    page = st.radio("NAVIGATION",[
        "🏠  Home",
        "📊  Data Studio",
        "🔬  Model Lab",
        "📈  Evaluation Suite",
        "🧠  Model Optimizer",
        "📋  Experiment Tracker",
    ])

    st.markdown("---")
    st.markdown(f"<div style='color:{MUTED};font-size:11px;'>SESSION STATE</div>",
                unsafe_allow_html=True)
    has_data  = st.session_state.get("dataset") is not None
    has_model = len(st.session_state.get("trained_models",{})) > 0
    st.markdown(f"""
    <div style='font-size:12px;margin-top:8px;'>
        {'<span class="badge badge-green">✓ Dataset</span>' if has_data else '<span style="color:#f85149;font-size:12px;">✗ No dataset</span>'}
        <br/><br/>
        {'<span class="badge badge-green">✓ Model trained</span>' if has_model else '<span style="color:#f85149;font-size:12px;">✗ No model</span>'}
        <br/><br/>
        {'<span class="badge badge-blue">Optuna ✓</span>' if HAS_OPTUNA else '<span style="color:#d29922;font-size:12px;">⚠ Optuna missing</span>'}
        {'&nbsp;<span class="badge badge-blue">XGBoost ✓</span>' if HAS_XGB else ''}
        {'&nbsp;<span class="badge badge-blue">LGB ✓</span>' if HAS_LGB else ''}
    </div>""",unsafe_allow_html=True)

# Route pages
if   "Home"        in page: page_home()
elif "Data Studio" in page: page_data()
elif "Model Lab"   in page: page_model_lab()
elif "Evaluation"  in page: page_eval()
elif "Optimizer"   in page: page_optimizer()
elif "Experiment"  in page: page_experiments()