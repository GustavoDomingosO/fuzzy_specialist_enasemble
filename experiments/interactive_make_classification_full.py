"""
Streamlit app to interactively generate and visualize sklearn.datasets.make_classification,
plus model comparison using 5-fold CV (Accuracy + G-mean).

Run:
    pip install streamlit scikit-learn matplotlib pandas numpy imbalanced-learn
    streamlit run interactive_make_classification_full.py
"""

import io
import sys
from typing import Optional, List, Dict

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score
from imblearn.metrics import geometric_mean_score

# Proposed Fuzzy Models
from Final_Models.PipelineEight_BASE import FinalFuzzyEnsembleClassifier

# --- NOVAS FUNÇÕES PARA TESTES ESPECÍFICOS ---

def apply_skew_and_outliers(X: np.ndarray, skew_factor: float, outlier_ratio: float, random_state: int):
    """Transforma dados normais em assimétricos e adiciona outliers."""
    # Create a local random number generator tied to your seed
    rng = np.random.RandomState(random_state)
    
    # Aplicar assimetria (Exponencial)
    if skew_factor > 0:
        X = np.exp(X * skew_factor)
    
    # Adicionar Outliers (Valores extremos aleatórios)
    if outlier_ratio > 0:
        n_outliers = int(len(X) * outlier_ratio)
        # Use the seeded rng instead of np.random
        outlier_indices = rng.choice(len(X), n_outliers, replace=False)
        X[outlier_indices] *= rng.uniform(5, 10, size=(n_outliers, X.shape[1]))
        
    return X

def generate_spatial_rule_change(params: Dict):
    """Gera dois datasets com regras diferentes e os combina (estilo GAMETES/Local Rules)."""
    # Dataset 1: Regra Original
    X1, y1 = make_classification(**params)
    
    # Dataset 2: Regra Invertida ou Diferente (mudando a semente e flip)
    params_alt = params.copy()
    params_alt['random_state'] += 100
    X2, y2 = make_classification(**params_alt)
    
    # Simula mudança de região: Se a primeira feature for > 0, usa regra 1, senão regra 2
    # Isso cria uma descontinuidade lógica no espaço de decisão
    mask = X1[:, 0] > 0
    X_combined = np.where(mask[:, None], X1, X2)
    y_combined = np.where(mask, y1, y2)
    
    return X_combined, y_combined

# ---------------------------------------------

st.set_page_config(page_title="make_classification visualizer", layout="wide")

st.title("Interactive `make_classification` visualizer")

# ---------------- Sidebar ----------------
st.sidebar.header("Dataset parameters")

n_samples = st.sidebar.slider("n_samples", 50, 20000, 1500, step=10)
n_features = st.sidebar.slider("n_features", 2, 200, 15)
n_informative = st.sidebar.slider("n_informative", 1, max(1, n_features), min(5, n_features))
n_redundant = st.sidebar.slider("n_redundant", 0, max(0, n_features - n_informative), 0)
n_clusters_per_class = st.sidebar.slider("n_clusters_per_class", 1, 10, 2)
class_sep = st.sidebar.slider("class_sep", 0.1, 10.0, 1.0)
flip_y = st.sidebar.slider("flip_y", 0.0, 0.5, 0.01)
n_classes = st.sidebar.slider("n_classes", 2, 10, 2)
random_state = st.sidebar.number_input("random_state", value=42)
st.sidebar.markdown("---")

st.sidebar.markdown("#### Configurações Visuais")
plot_title = st.sidebar.text_input("Título do Gráfico PCA", value="Projeção (PCA se atributos > 2)")
project_with_pca = st.sidebar.checkbox("Project to 2D with PCA (if features>2)", value=True)
show_decision_boundary = st.sidebar.checkbox("Show decision boundary (binary only)", value=False)

st.sidebar.header("🧪 Advanced Experiments")

# Toggle para Regras Espaciais
use_spatial_rules = st.sidebar.checkbox("Regras que mudam por região (Local Rules)", value=False)

# Sliders para Assimetria e Outliers
skew_factor = st.sidebar.slider("Assimetria (Skewness)", 0.0, 2.0, 0.0, help="0 = Normal (Gaussiana)")
outlier_ratio = st.sidebar.slider("Taxa de Outliers", 0.0, 0.2, 0.0)

# Novo: Controles de Escala
st.sidebar.markdown("#### Magnitude e Escala")
scale_factor = st.sidebar.selectbox(
    "Fator de Escala (Multiplicador)", 
    options=[1.0, 10.0, 100.0, 1000.0, 10000.0, 1000000.0], 
    index=0, 
    help="Multiplica os valores do dataset para simular magnitudes extremas."
)
partial_scale = st.sidebar.checkbox(
    "Aplicar escala em apenas 2 features (Desnível)", 
    value=False,
    help="Se marcado, apenas as duas primeiras colunas receberão a magnitude extrema."
)

# ---------------- Helpers ----------------

def parse_weights(weights_text: str, n_classes: int) -> Optional[List[float]]:
    text = (weights_text or "").strip()
    if not text:
        return None
    try:
        parts = [float(p) for p in text.split(",") if p.strip() != ""]
        if len(parts) != n_classes:
            return None
        s = sum(parts)
        if s <= 0:
            return None
        return [p / s for p in parts]
    except Exception:
        return None

def generate_dataset_advanced(params: Dict, use_spatial: bool, skew: float, outliers: float, scale: float, partial: bool, random_state: int):
    if use_spatial:
        X, y = generate_spatial_rule_change(params)
    else:
        X, y = make_classification(**params)
    
    # 1. Aplica transformações de distribuição (Assimetria/Outliers)
    if skew > 0 or outliers > 0:
        X = apply_skew_and_outliers(X, skew, outliers, random_state)
        
    # 2. Aplica transformação de Escala (Magnitude)
    if scale > 1.0:
        if partial and X.shape[1] >= 2:
            X[:, :2] *= scale  # Multiplica apenas as duas primeiras colunas
        else:
            X *= scale         # Multiplica todo o dataset
        
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df

def plot_projection(df: pd.DataFrame, project_with_pca: bool, show_boundary: bool, custom_title: str):
    X = df[[c for c in df.columns if c.startswith("f")]].values
    y = df["target"].values

    if X.shape[1] > 2 and project_with_pca:
        pca = PCA(n_components=2, random_state=random_state)
        X2 = pca.fit_transform(X)
    else:
        X2 = X[:, :2]

    fig, ax = plt.subplots(figsize=(6, 5))
    classes = np.unique(y)

    for cls in classes:
        mask = y == cls
        ax.scatter(X2[mask, 0], X2[mask, 1], label=f"classe {cls}", alpha=0.8)

    if show_boundary and len(classes) == 2:
        clf = LogisticRegression().fit(X2, y)
        x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
        y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.07)

    ax.set_title(custom_title)
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def evaluate_models(df: pd.DataFrame):
    X = df.drop(columns="target").values
    y = df["target"].values

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "Bagging": BaggingClassifier(random_state=random_state),
        "Model Proposto": FinalFuzzyEnsembleClassifier(random_state=random_state),
    }

    # Initialize the cross-validator
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Determine the correct average parameter for G-mean
    gmean_avg = "macro" if len(np.unique(y)) > 2 else "binary"

    results = []

    for name, model in models.items():
        fold_accuracies = []
        fold_gmeans = []

        # Manually iterate through each fold
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Direct fit and predict calls
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate and store metrics for the current fold
            fold_accuracies.append(accuracy_score(y_test, y_pred))
            fold_gmeans.append(geometric_mean_score(y_test, y_pred, average=gmean_avg))

        # Average the metrics and compute standard deviation across all 5 folds
        results.append({
            "modelo": name,
            "acurácia": np.mean(fold_accuracies),
            "acurácia_std": np.std(fold_accuracies), # Added Std
            "Média-G": np.mean(fold_gmeans),
            "Média-G_std": np.std(fold_gmeans),      # Added Std
        })

    return pd.DataFrame(results)

def plot_model_comparison(results_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))

    x = np.arange(len(results_df))
    width = 0.35

    # Added yerr and capsize for error bars
    ax.bar(x - width/2, results_df["acurácia"], width, 
           yerr=results_df["acurácia_std"], capsize=5, label="Acurácia")
    
    ax.bar(x + width/2, results_df["Média-G"], width, 
           yerr=results_df["Média-G_std"], capsize=5, label="Média-G")

    ax.set_xticks(x)
    ax.set_xticklabels(results_df["modelo"])
    ax.set_ylim(0, 1.1) # Increased limit slightly to accommodate error bars at the top
    ax.set_title("5-fold CV: Acurácia vs Média-G")
    ax.legend()
    ax.grid(alpha=0.2)

    return fig


# ---------------- Dataset ----------------

weights = None # Lógica simplificada para o exemplo
params = dict(
    n_samples=n_samples, n_features=n_features, n_informative=n_informative,
    n_redundant=n_redundant, n_clusters_per_class=n_clusters_per_class,
    class_sep=class_sep, flip_y=flip_y, n_classes=n_classes,
    random_state=random_state, weights=weights
)

df = generate_dataset_advanced(params, use_spatial_rules, skew_factor, outlier_ratio, scale_factor, partial_scale, random_state)

# ---------------- Layout ----------------

col1, col2 = st.columns([2, 1])

with col1:
    # Agora passamos o título customizado configurado na sidebar
    st.pyplot(plot_projection(df, project_with_pca, show_decision_boundary, plot_title))
    st.markdown("### Model Comparison (5-fold CV)")
    results_df = evaluate_models(df)
    st.pyplot(plot_model_comparison(results_df))

with col2:
    st.markdown("### Dataset stats")
    st.table(df["target"].value_counts().sort_index())

    st.markdown("### CV Results")
    st.dataframe(results_df)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="make_classification.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown("Models evaluated with Stratified 5-fold cross-validation.")