
# app_streamlit_clusters.py
# -------------------------------------------------------------
# Plataforma unificada para: Clustering, Clasificación y Regresión
# - Carga de archivos (CSV, Excel, JSON)
# - Preprocesamiento: selección de variables, imputación, escalado
# - Manejo de outliers: automático y manual
# - Clustering: KMeans, GMM, Agglomerative, Spectral, DBSCAN, HDBSCAN
# - Visualizaciones 2D/3D con PCA, t-SNE y Convex Hull
# - Clasificación: KNN, Regresión Logística, Árbol
# - Regresión: Lineal, Ridge, Lasso, KNN
# -------------------------------------------------------------

from __future__ import annotations

import io
import json
from typing import Dict, List, Optional, Tuple, Iterable, Set

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from scipy.spatial import ConvexHull

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
    DBSCAN,
)
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    confusion_matrix,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# Optional HDBSCAN support
try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# =========================
# Shared utilities
# =========================

def read_any(file) -> pd.DataFrame:
    """Read a CSV, Excel, or JSON file into a DataFrame."""
    name = file.name.lower()
    data = file.read()

    if name.endswith(".csv") or (b"," in data[:100]):
        for enc in ("utf-8", "latin-1"):
            try:
                return pd.read_csv(
                    io.BytesIO(data),
                    encoding=enc,
                )
            except Exception:
                pass

    if name.endswith((".xlsx", ".xls", ".xlsm")):
        try:
            return pd.read_excel(io.BytesIO(data))
        except Exception as e:
            st.warning(
                f"No se pudo leer Excel: {e}"
            )

    try:
        obj = json.loads(data.decode(errors="ignore"))

        if isinstance(obj, (list, dict)):
            return pd.DataFrame(obj)

    except Exception:
        pass

    try:
        return pd.read_table(io.BytesIO(data))
    except Exception as e:
        raise ValueError(
            "Formato de archivo no soportado o corrupto"
        ) from e


def split_features_target(
    df: pd.DataFrame,
    target_col: Optional[str],
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y

    return df, None


def select_numeric(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]


def build_preprocessor(
    num_cols: List[str],
    scale_mode: str,
    impute_strategy: str,
) -> ColumnTransformer:

    if scale_mode == "Estandar (media=0, sd=1)":
        scaler = StandardScaler()

    elif scale_mode == "MinMax [0,1]":
        scaler = MinMaxScaler()

    elif scale_mode == "Robusto (mediana/IQR)":
        scaler = RobustScaler()

    else:
        scaler = "passthrough"

    imputer = SimpleImputer(
        strategy=(
            "median"
            if impute_strategy == "Mediana"
            else "mean"
        )
    )

    numeric_pipe = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
    ])

    return ColumnTransformer(
        [("num", numeric_pipe, num_cols)],
        remainder="drop",
    )


def plot_corr(
    df: pd.DataFrame,
    title: str,
    key: Optional[str] = None,
):
    if df.empty:
        st.info("No hay datos para correlación.")
        return

    corr = df.corr(numeric_only=True)

    if corr.empty:
        st.info("No hay correlaciones disponibles.")
        return

    fig = ff.create_annotated_heatmap(
        corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        showscale=True,
        colorscale="Viridis",
    )

    fig.update_layout(
        title=title,
        height=600,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key or f"corr_{abs(hash(title)) % 10_000_000}",
    )


# =========================
# Classification models
# =========================

def build_classifier(name: str, params: Dict) -> object:

    if name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=params.get("k", 5),
            weights=params.get("weights", "uniform"),
            p=params.get("p", 2),
        )

    if name == "LogisticRegression":
        return LogisticRegression(max_iter=500)

    if name == "DecisionTree":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", None),
            random_state=42,
        )

    return KNeighborsClassifier()


# =========================
# Regression models
# =========================

def build_regressor(name: str, params: Dict) -> object:

    if name == "Linear":
        return LinearRegression()

    if name == "Ridge":
        return Ridge(
            alpha=params.get("alpha", 1.0)
        )

    if name == "Lasso":
        return Lasso(
            alpha=params.get("alpha", 0.01)
        )

    if name == "KNN":
        return KNeighborsRegressor(
            n_neighbors=params.get("k", 5),
            weights=params.get("weights", "uniform"),
            p=params.get("p", 2),
        )

    return LinearRegression()


# =========================
# Clustering metrics
# =========================

def cluster_internal_metrics(
    X_num: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:

    out: Dict[str, float] = {}

    try:
        uniq = np.unique(labels)

        if len(uniq) > 1 and X_num.shape[0] > len(uniq):
            out["Silhouette"] = float(
                silhouette_score(X_num, labels)
            )

    except Exception:
        pass

    try:
        if len(np.unique(labels)) > 1:
            out["Calinski-Harabasz"] = float(
                calinski_harabasz_score(X_num, labels)
            )

            out["Davies-Bouldin"] = float(
                davies_bouldin_score(X_num, labels)
            )

    except Exception:
        pass

    return out


# =========================
# Visualization: Convex Hulls
# =========================

def _draw_hulls_2d(
    fig,
    emb2d: np.ndarray,
    labels: np.ndarray,
    include: Iterable[int],
    opacity: float = 0.12,
    name_map: Optional[Dict[int, str]] = None,
):
    include_set = set(include)
    uniq = [
        cl for cl in sorted(np.unique(labels))
        if cl in include_set
    ]

    for cl in uniq:
        pts = emb2d[labels == cl]

        if pts.shape[0] >= 3:
            try:
                hull = ConvexHull(pts)
                verts = pts[hull.vertices]

                x = np.r_[verts[:, 0], verts[0, 0]]
                y = np.r_[verts[:, 1], verts[0, 1]]

                label_name = (
                    name_map.get(cl, f"Hull {cl}")
                    if name_map
                    else f"Hull {cl}"
                )

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=label_name,
                        fill="toself",
                        opacity=opacity,
                        line=dict(width=1),
                    )
                )

            except Exception:
                continue

    return fig


def _draw_hulls_3d(
    fig,
    emb3d: np.ndarray,
    labels: np.ndarray,
    include: Iterable[int],
    opacity: float = 0.15,
    name_map: Optional[Dict[int, str]] = None,
):
    include_set = set(include)

    uniq = [
        cl for cl in sorted(np.unique(labels))
        if cl in include_set
    ]

    for cl in uniq:
        pts = emb3d[labels == cl]

        if pts.shape[0] >= 4:
            try:
                hull = ConvexHull(pts)
                i, j, k = hull.simplices.T

                label_name = (
                    name_map.get(cl, f"Hull {cl}")
                    if name_map
                    else f"Hull {cl}"
                )

                fig.add_trace(
                    go.Mesh3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        i=i,
                        j=j,
                        k=k,
                        name=label_name,
                        opacity=opacity,
                        alphahull=0,
                        showscale=False,
                    )
                )

            except Exception:
                continue

    return fig


# =========================
# Fixed-space visualization
# =========================

def scatter_embed_fixed_space(
    emb: np.ndarray,
    labels_all: np.ndarray,
    mask_visible: np.ndarray,
    dim: int,
    title: str,
    show_hull: bool,
    hull_include_set: Optional[Iterable[int]],
    plot_key: Optional[str] = None,
):
    """
    Use embeddings calculated from all observations.

    Hiding clusters does not change point positions.
    """

    emb_vis = emb[mask_visible]
    labels_vis = labels_all[mask_visible]

    name_map = {-1: "Hull outliers"}

    if dim == 2:
        df_plot = pd.DataFrame({
            "x": emb_vis[:, 0],
            "y": emb_vis[:, 1],
            "cluster": labels_vis.astype(str),
        })

        fig = px.scatter(
            df_plot,
            x="x",
            y="y",
            color="cluster",
            title=title,
            opacity=0.85,
        )

        if show_hull and hull_include_set:
            include_here = (
                set(hull_include_set)
                & set(np.unique(labels_vis))
            )

            fig = _draw_hulls_2d(
                fig,
                emb_vis,
                labels_vis,
                include=include_here,
                opacity=0.12,
                name_map=name_map,
            )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=plot_key or (
                f"scatter2d_fixed_"
                f"{abs(hash(title)) % 10_000_000}"
            ),
        )

    else:
        df3 = pd.DataFrame({
            "x": emb_vis[:, 0],
            "y": emb_vis[:, 1],
            "z": emb_vis[:, 2],
            "cluster": labels_vis.astype(str),
        })

        fig3 = px.scatter_3d(
            df3,
            x="x",
            y="y",
            z="z",
            color="cluster",
            title=title,
            opacity=0.85,
        )

        if show_hull and hull_include_set:
            include_here = (
                set(hull_include_set)
                & set(np.unique(labels_vis))
            )

            fig3 = _draw_hulls_3d(
                fig3,
                emb_vis,
                labels_vis,
                include=include_here,
                opacity=0.15,
                name_map=name_map,
            )

        st.plotly_chart(
            fig3,
            use_container_width=True,
            key=plot_key or (
                f"scatter3d_fixed_"
                f"{abs(hash(title)) % 10_000_000}"
            ),
        )


# =========================
# Automatic outlier detection
# =========================

def detect_outliers_auto(
    X_num: np.ndarray,
    method: str = "IsolationForest",
    contamination: float = 0.05,
    z_thr: float = 3.5,
) -> np.ndarray:

    n = X_num.shape[0]
    is_out = np.zeros(n, dtype=bool)

   
    if method == "IsolationForest":
        if contamination <= 0:
            return is_out

        try:
            model = IsolationForest(
                contamination=float(contamination),
                random_state=42,
            )

            is_out = model.fit_predict(X_num) == -1

        except Exception as e:
            st.error(
                f"Error al detectar outliers con "
                f"IsolationForest: {e}"
            )


        except Exception:
            pass

    else:
        # Robust Z-score using Median Absolute Deviation
        med = np.median(X_num, axis=0)

        mad = (
            np.median(
                np.abs(X_num - med),
                axis=0,
            )
            + 1e-12
        )

        zrob = (
            0.6745
            * (X_num - med)
            / mad
        )

        is_out = (
            np.abs(zrob) > float(z_thr)
        ).any(axis=1)

    return is_out


# =========================
# Manual exclusion helpers
# =========================

def _ensure_excluded_list(n: int) -> None:

    
    if "manual_excluded_idx" not in st.session_state:
        st.session_state["manual_excluded_idx"] = []

    if "active_dataset_id" not in st.session_state:
        st.session_state["active_dataset_id"] = None


    st.session_state["manual_excluded_idx"] = sorted({
        int(i)
        for i in st.session_state["manual_excluded_idx"]
        if 0 <= int(i) < n
    })


def _parse_indices_text(s: str) -> List[int]:

    if not s:
        return []

    tmp = (
        s.replace(";", ",")
        .replace("\n", ",")
    )

    out = []

    for tok in tmp.split(","):
        tok = tok.strip()

        if tok == "":
            continue

        out.append(int(tok))

    return out


def exclude_indices(
    idxs: Iterable[int],
    n: int,
) -> None:

    _ensure_excluded_list(n)

    safe = {
        int(i)
        for i in idxs
        if 0 <= int(i) < n
    }

    merged = (
        set(
            map(
                int,
                st.session_state["manual_excluded_idx"],
            )
        )
        | safe
    )

    st.session_state["manual_excluded_idx"] = sorted(
        merged
    )


def include_indices(
    idxs: Iterable[int],
    n: int,
) -> None:

    _ensure_excluded_list(n)

    rem = {
        int(i)
        for i in idxs
    }

    st.session_state["manual_excluded_idx"] = [
        int(i)
        for i in st.session_state["manual_excluded_idx"]
        if int(i) not in rem
    ]


def exclude_point(
    idx: int,
    n: int,
) -> None:
    """Exclude a single observation by index."""

    exclude_indices([idx], n)


# =========================
# Cluster explainability
# =========================

def explain_clusters_numeric(
    df: pd.DataFrame,
    used_num: List[str],
    labels_all: np.ndarray,
    noise_label: int = -1,
) -> Dict:
    """
    Explain clusters using standardized differences
    between cluster means and global means.
    """

    res: Dict[int, Dict] = {}

    if len(used_num) == 0:
        return res

    num_df = df[used_num].copy()

    global_stats = num_df.describe(
        percentiles=[0.25, 0.5, 0.75]
    ).T

    for cl in sorted(np.unique(labels_all)):

        if cl == noise_label:
            continue

        idx = np.where(
            labels_all == cl
        )[0]

        if len(idx) == 0:
            continue

        sub = num_df.iloc[idx]

        stats = sub.describe(
            percentiles=[0.25, 0.5, 0.75]
        ).T

        denom = (
            global_stats["std"]
            .replace(0, np.nan)
            .fillna(1e-9)
        )

        lift = (
            stats["mean"] - global_stats["mean"]
        ) / (denom + 1e-12)

        top_idx = (
            lift.abs()
            .sort_values(ascending=False)
            .head(5)
            .index
        )

        res[int(cl)] = {
            "n": int(len(sub)),
            "top_features_by_lift": [
                {
                    "feature": f,
                    "lift_z": (
                        float(lift.loc[f])
                        if f in lift.index
                        else None
                    ),
                    "mean": (
                        float(stats.loc[f, "mean"])
                        if f in stats.index
                        else None
                    ),
                    "median": (
                        float(stats.loc[f, "50%"])
                        if f in stats.index
                        else None
                    ),
                }
                for f in top_idx
            ],
        }

    return res



def explain_clusters_logreg(
    X_num_keep: np.ndarray,
    labels_fit: np.ndarray,
    used_num: List[str],
) -> Optional[Dict]:

    uniq = np.unique(labels_fit)

    if len(uniq) < 2:
        return None

    try:
        lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

        lr.fit(X_num_keep, labels_fit)

        coefs = lr.coef_
        classes = list(lr.classes_)
        mapping = {}

        # Binary classification returns one coefficient row.
        # The first class uses the opposite coefficients.
        if len(classes) == 2 and coefs.shape[0] == 1:
            coefs = np.vstack([
                -coefs[0],
                coefs[0],
            ])

        for i, cl in enumerate(classes):
            coef = coefs[i]

            idx = np.argsort(
                np.abs(coef)
            )[-5:][::-1]

            mapping[int(cl)] = [
                {
                    "feature": used_num[j],
                    "coef": float(coef[j]),
                }
                for j in idx
            ]

        return mapping

    except Exception:
        return None


# =========================
# Cluster data preview
# =========================

def preview_clusters_ui(
    df: pd.DataFrame,
    labels_all: np.ndarray,
    used_num: List[str],
    all_cols: List[str],
    visible_clusters: Iterable[int],
):

    st.subheader(
        "Previsualización de datos por cluster"
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns(
        [1, 1, 1]
    )

    with col_cfg1:
        only_used = st.checkbox(
            "Mostrar solo columnas usadas (X)",
            value=True,
            key="pv_only_used",
        )

    with col_cfg2:
        head_n = st.slider(
            "Filas por cluster (vista)",
            min_value=5,
            max_value=200,
            value=20,
            step=5,
            key="pv_head_n",
        )

    with col_cfg3:
        only_visible = st.checkbox(
            "Solo clusters visibles",
            value=True,
            key="pv_only_visible",
        )

    cluster_ids = sorted([
        int(c)
        for c in np.unique(labels_all)
    ])

    if only_visible:
        visible_set = set(visible_clusters)

        cluster_ids = [
            c for c in cluster_ids
            if c in visible_set
        ]

    tab_labels = [
        (
            "Outliers (-1)"
            if c < 0
            else f"Cluster {c}"
        )
        for c in cluster_ids
    ]

    if len(cluster_ids) == 0:
        st.info(
            "No hay clusters para previsualizar "
            "con la configuración actual."
        )
        return

    tabs = st.tabs(tab_labels)
    total_n = len(df)

    for tab, c in zip(tabs, cluster_ids):

        with tab:
            idx = np.where(
                labels_all == c
            )[0]

            n_c = int(len(idx))

            st.write({
                "cluster": int(c),
                "n": n_c,
                "% del total": round(
                    100.0 * n_c / max(1, total_n),
                    2,
                ),
            })

            if n_c == 0:
                st.info(
                    "Sin filas en este cluster."
                )
                continue

            df_sub = df.iloc[idx].copy()

            if only_used:
                view_cols = list(used_num)
            else:
                view_cols = list(all_cols)

            st.dataframe(
                df_sub[view_cols].head(head_n),
                use_container_width=True,
            )

            with st.expander(
                "Resumen numérico (describe)"
            ):

                if used_num:
                    st.dataframe(
                        df_sub[used_num].describe().T,
                        use_container_width=True,
                    )

                else:
                    st.caption(
                        "No hay columnas numéricas seleccionadas."
                    )


# =====================================================
# CLUSTERING TASK
# =====================================================

def run_clustering_task(
    df: pd.DataFrame,
    used_num: List[str],
    all_cols: List[str],
    scale_mode: str,
    impute_strategy: str,
    target_col_global: Optional[str],
):
    st.header("Clustering")

    with st.sidebar:
        st.subheader("Parámetros de Clustering")

        algo = st.selectbox(
            "Algoritmo",
            [
                "KMeans",
                "GaussianMixture",
                "Agglomerative",
                "Spectral",
                "DBSCAN",
            ] + (["HDBSCAN"] if HDBSCAN_AVAILABLE else []),
            key="cl_algo",
        )

        auto_k = st.checkbox(
            "Elegir k automáticamente (Silhouette)",
            value=algo in [
                "KMeans",
                "GaussianMixture",
                "Agglomerative",
            ],
            key="cl_auto_k",
        )

        k_min, k_max = 2, 10

        if auto_k:
            k_min, k_max = st.slider(
                "Rango de k",
                min_value=2,
                max_value=20,
                value=(2, 10),
                key="cl_k_range",
            )

        n_clusters = st.number_input(
            "n_clusters (si aplica)",
            min_value=2,
            max_value=50,
            value=4,
            key="cl_k",
        )

        eps = st.number_input(
            "eps (DBSCAN)",
            min_value=0.01,
            max_value=10.0,
            value=0.5,
            step=0.01,
            key="cl_eps",
        )

        min_samples = st.number_input(
            "min_samples (DBSCAN/HDBSCAN)",
            min_value=2,
            max_value=200,
            value=10,
            key="cl_min_samples",
        )

        dim_red = st.selectbox(
            "Proyección para gráfica",
            ["PCA", "TSNE"],
            index=0,
            key="cl_dimred",
        )

        plot_dim = st.selectbox(
            "Dimensiones de gráfica",
            ["2D", "3D"],
            index=0,
            key="cl_dim",
        )

        mostrar_hull = st.checkbox(
            "Mostrar convex hull",
            value=True,
            key="cl_show_hull",
        )

        # -----------------------------------------
        # Outlier configuration
        # -----------------------------------------

        st.subheader("Manejo de Outliers")

        out_mode = st.selectbox(
            "Modo",
            ["Ninguno", "Automático", "Manual"],
            index=0,
            key="cl_out_mode",
        )

        # Retained for UI compatibility.
        # Excluded observations are currently assigned -1.
        exclude_as_noise = st.checkbox(
            "Etiquetar excluidos/outliers como 'ruido' (-1)",
            value=True,
            key="cl_excl_noise",
        )

        out_method = "IsolationForest"
        contamination = 0.05
        z_thr = 3.5

        if out_mode == "Automático":
            out_method = st.selectbox(
                "Método",
                [
                    "IsolationForest",
                    "Zscore robusto (MAD)",
                ],
                key="cl_out_method",
            )

            if out_method == "IsolationForest":
                contamination = st.slider(
                    "Contaminación esperada",
                    0.0,
                    0.3,
                    0.05,
                    0.01,
                    key="cl_out_cont",
                )
            else:
                z_thr = st.number_input(
                    "Umbral |z_rob|",
                    min_value=2.0,
                    max_value=10.0,
                    value=3.5,
                    step=0.1,
                    key="cl_out_zthr",
                )

        # -----------------------------------------
        # Section visibility
        # -----------------------------------------

        st.subheader("Visibilidad de secciones")

        v_show_menu_clusters = st.checkbox(
            "Menú de clusters (visibilidad/hulls)",
            True,
            key="v_show_menu_clusters",
        )

        v_show_visual = st.checkbox(
            "Visualización (scatter 2D/3D)",
            True,
            key="v_show_visual",
        )

        v_show_preview = st.checkbox(
            "Previsualización por cluster",
            True,
            key="v_show_preview",
        )

        v_show_internal = st.checkbox(
            "Métricas internas",
            True,
            key="v_show_internal",
        )

        v_show_corr_global = st.checkbox(
            "Correlación global",
            True,
            key="v_show_corr_global",
        )

        v_show_corr_per_cluster = st.checkbox(
            "Correlación por cluster",
            False,
            key="v_show_corr_per_cluster",
        )

        v_show_explanation = st.checkbox(
            "Explicación (stats + LR)",
            True,
            key="v_show_explanation",
        )

        v_show_thresholds = st.checkbox(
            "Porcentajes bajo/encima de umbral",
            False,
            key="v_show_thresholds",
        )

        v_show_supervised_block = st.checkbox(
            "Entrenar modelo de REGRESIÓN por cluster",
            False,
            key="v_show_supervised_block",
        )

        v_show_export_col = st.checkbox(
            "Sección de exportación por cluster",
            True,
            key="v_show_export_col",
        )

    # =========================================
    # Preprocessing
    # =========================================

    if not used_num:
        st.error(
            "Selecciona al menos una columna numérica."
        )
        st.stop()

    pre = build_preprocessor(
        used_num,
        scale_mode,
        impute_strategy,
    )

    X_pre = pre.fit_transform(df)
    X_num = np.asarray(X_pre)

    n = X_num.shape[0]

    if "manual_excluded_idx" not in st.session_state:
        st.session_state["manual_excluded_idx"] = []

    # =========================================
    # Outlier detection
    # =========================================

    manual_mask = np.zeros(n, dtype=bool)

    if st.session_state["manual_excluded_idx"]:
        safe_idx = np.array(
            st.session_state["manual_excluded_idx"],
            dtype=int,
        )

        safe_idx = safe_idx[
            (safe_idx >= 0) & (safe_idx < n)
        ]

        manual_mask[safe_idx] = True

    auto_mask = np.zeros(n, dtype=bool)

    if out_mode == "Automático":
        auto_mask = detect_outliers_auto(
            X_num,
            method=out_method,
            contamination=contamination,
            z_thr=z_thr,
        )

        st.write({
            "outliers_detectados": int(auto_mask.sum())
        })

    mask_exclude = manual_mask | auto_mask
    keep_idx = np.where(~mask_exclude)[0]

    if len(keep_idx) < 5:
        st.error(
            "Demasiados puntos excluidos. "
            "Reduce el filtro de outliers o "
            "restablece la selección manual."
        )
        st.stop()

    # =========================================
    # Automatic cluster selection
    # =========================================

    chosen_k = int(n_clusters)

    if auto_k and algo in (
        "KMeans",
        "GaussianMixture",
        "Agglomerative",
    ):
        best_score = -1.0
        best_k = None

        for k in range(k_min, k_max + 1):
            if k >= len(keep_idx):
                continue

            try:
                if algo == "KMeans":
                    labels_try = KMeans(
                        n_clusters=k,
                        random_state=42,
                        n_init="auto",
                    ).fit_predict(X_num[keep_idx])

                elif algo == "GaussianMixture":
                    gmm_try = GaussianMixture(
                        n_components=k,
                        random_state=42,
                    )

                    labels_try = gmm_try.fit_predict(
                        X_num[keep_idx]
                    )

                else:
                    labels_try = AgglomerativeClustering(
                        n_clusters=k
                    ).fit_predict(X_num[keep_idx])

                score = silhouette_score(
                    X_num[keep_idx],
                    labels_try,
                )

                if score > best_score:
                    best_score = score
                    best_k = k

            except Exception:
                continue

        if best_k is not None:
            chosen_k = int(best_k)

            st.success(
                f"k óptimo (Silhouette) ≈ {best_k} | "
                f"Silhouette={best_score:.3f}"
            )
        else:
            st.warning(
                "No se pudo estimar k automáticamente; "
                "usando n_clusters indicado."
            )

    # =========================================
    # Model training
    # =========================================

    
    # Validate clustering parameters before model training.
    n_available = len(keep_idx)

    if algo in (
        "KMeans",
        "GaussianMixture",
        "Agglomerative",
        "Spectral",
    ):
        if chosen_k < 2:
            st.error(
                "El número de clusters debe ser al menos 2."
            )
            st.stop()

        if chosen_k > n_available:
            st.error(
                f"Solicitaste {chosen_k} clusters, "
                f"pero solo hay {n_available} observaciones "
                "disponibles después de excluir outliers."
            )
            st.stop()

    if algo == "Spectral":
        if n_available < 3:
            st.error(
                "SpectralClustering requiere al menos "
                "3 observaciones para esta configuración."
            )
            st.stop()

        if n_available <= 10:
            st.warning(
                "SpectralClustering puede producir resultados "
                "inestables con datasets muy pequeños."
            )

    if algo in ("DBSCAN", "HDBSCAN"):
        if int(min_samples) > n_available:
            st.warning(
                f"min_samples={min_samples} supera las "
                f"{n_available} observaciones disponibles. "
                "Es posible que todos los puntos sean "
                "clasificados como ruido."
            )


    if algo == "KMeans":
        model = KMeans(
            n_clusters=chosen_k,
            random_state=42,
            n_init="auto",
        )

        labels_fit = model.fit_predict(
            X_num[keep_idx]
        )

    elif algo == "GaussianMixture":
        model = GaussianMixture(
            n_components=chosen_k,
            random_state=42,
        )

        labels_fit = model.fit_predict(
            X_num[keep_idx]
        )

    elif algo == "Agglomerative":
        model = AgglomerativeClustering(
            n_clusters=chosen_k
        )

        labels_fit = model.fit_predict(
            X_num[keep_idx]
        )

    elif algo == "Spectral":
        n_neighbors = min(
            10,
            len(keep_idx) - 1,
        )

        model = SpectralClustering(
            n_clusters=chosen_k,
            random_state=42,
            assign_labels="kmeans",
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
        )

        labels_fit = model.fit_predict(
            X_num[keep_idx]
        )


    elif algo == "DBSCAN":
        model = DBSCAN(
            eps=float(eps),
            min_samples=int(min_samples),
        )

        labels_fit = model.fit_predict(
            X_num[keep_idx]
        )

    else:
        model = hdbscan.HDBSCAN(
            min_samples=int(min_samples)
        )

        labels_fit = model.fit_predict(
            X_num[keep_idx]
        )

    # =========================================
    # Restore labels to original dataset
    # =========================================

    # Excluded observations are marked as noise (-1).
    # This preserves the original behavior.
    labels_all = np.full(n, -1, dtype=int)
    labels_all[keep_idx] = labels_fit

    # =========================================
    # Cluster visibility and export controls
    # =========================================

    visible_clusters = None
    hull_include: Set[int] = set()

    include_outliers_label = (
        -1 in np.unique(labels_all)
    )

    if v_show_menu_clusters:
        st.markdown("### Menú de clusters")

        unique_labels_all = sorted(
            np.unique(labels_all).tolist()
        )

        clusters_no_out = [
            int(c)
            for c in unique_labels_all
            if c >= 0
        ]

        colA, colB, colC = st.columns(
            [1.2, 1.2, 1.2]
        )

        with colA:
            st.markdown(
                "**Visibilidad de clusters "
                "(no altera posiciones)**"
            )

            show_outliers_pts = st.checkbox(
                "Mostrar outliers (-1)",
                value=False,
                key="show_outliers_pts",
            )

            options_vis = clusters_no_out + (
                [-1]
                if (
                    show_outliers_pts
                    and include_outliers_label
                )
                else []
            )

            prev_vis = st.session_state.get(
                "visible_clusters",
                clusters_no_out,
            )

            default_vis = [
                int(c)
                for c in prev_vis
                if c in options_vis
            ]

            if not default_vis and clusters_no_out:
                default_vis = clusters_no_out.copy()

            st.session_state["visible_clusters"] = (
                st.multiselect(
                    "Clusters visibles",
                    options=options_vis,
                    default=default_vis,
                    key="visible_clusters_ms",
                )
            )

        with colB:
            st.markdown("**Convex Hull**")

            options_hull = clusters_no_out

            prev_hulls = st.session_state.get(
                "selected_hull_clusters",
                clusters_no_out,
            )

            default_hulls = [
                int(c)
                for c in prev_hulls
                if c in options_hull
            ]

            if not default_hulls:
                default_hulls = []

            st.session_state[
                "selected_hull_clusters"
            ] = st.multiselect(
                "Hulls para clusters",
                options=options_hull,
                default=default_hulls,
                key="hull_clusters_ms",
            )

            st.session_state["hull_for_outliers"] = (
                st.checkbox(
                    "Dibujar hull para outliers (-1)",
                    value=False,
                    key="hull_outliers_ck",
                )
            )

        with colC:
            if v_show_export_col:
                st.markdown(
                    "**Exportación por cluster**"
                )

                cl_to_export = st.selectbox(
                    "Cluster a exportar",
                    options=unique_labels_all,
                    index=0,
                    key="exp_cluster_sel",
                )

                cols_export_mode = st.radio(
                    "Columnas a exportar",
                    ["Solo usadas (X)", "Todas"],
                    index=0,
                    horizontal=True,
                    key="exp_cols_mode",
                )

                rows_idx = np.where(
                    labels_all == cl_to_export
                )[0]

                if cols_export_mode == "Solo usadas (X)":
                    df_exp = df.iloc[rows_idx][used_num]
                else:
                    df_exp = df.iloc[rows_idx]

                csv_bytes = df_exp.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label=f"Descargar cluster_{cl_to_export}.csv",
                    data=csv_bytes,
                    file_name=f"cluster_{cl_to_export}.csv",
                    mime="text/csv",
                    key=f"dl_cluster_{cl_to_export}",
                )

        visible_clusters = st.session_state.get(
            "visible_clusters",
            clusters_no_out,
        )

        hull_include = set(
            st.session_state.get(
                "selected_hull_clusters",
                [],
            )
        )

        if (
            st.session_state.get(
                "hull_for_outliers",
                False,
            )
            and include_outliers_label
        ):
            hull_include.add(-1)

    else:
        unique_labels_all = sorted(
            np.unique(labels_all).tolist()
        )

        visible_clusters = [
            int(c)
            for c in unique_labels_all
            if c >= 0
        ]

        hull_include = set(visible_clusters)

    mask_visible = np.isin(
        labels_all,
        visible_clusters,
    )

    # =========================================
    # Visualization
    # =========================================

    if v_show_visual:
        st.subheader("Visualización")

        n_feats = len(used_num)

        if plot_dim == "3D" and n_feats < 3:
            st.error(
                "Para 3D necesitas al menos "
                "3 columnas numéricas."
            )
            st.stop()

        if plot_dim == "2D" and n_feats < 2:
            st.error(
                "Para 2D necesitas al menos "
                "2 columnas numéricas."
            )
            st.stop()

        if plot_dim == "2D":
            if dim_red == "PCA":
                emb_all = PCA(
                    n_components=2,
                    random_state=42,
                ).fit_transform(X_num)

            else:
               
                n_samples = X_num.shape[0]

                perplexity = min(
                    30.0,
                    max(2.0, n_samples / 20.0),
                    float(n_samples - 1),
                )

                emb_all = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=perplexity,
                ).fit_transform(X_num)


            scatter_embed_fixed_space(
                emb=emb_all,
                labels_all=labels_all,
                mask_visible=mask_visible,
                dim=2,
                title=(
                    f"{algo} — {dim_red} 2D | "
                    f"visibles={sorted(set(visible_clusters))}"
                ),
                show_hull=(
                    bool(hull_include)
                    and mostrar_hull
                ),
                hull_include_set=hull_include,
                plot_key="scatter_main_2d_fixed",
            )

        else:
            emb_all = PCA(
                n_components=3,
                random_state=42,
            ).fit_transform(X_num)

            scatter_embed_fixed_space(
                emb=emb_all,
                labels_all=labels_all,
                mask_visible=mask_visible,
                dim=3,
                title=(
                    f"{algo} — PCA 3D | "
                    f"visibles={sorted(set(visible_clusters))}"
                ),
                show_hull=(
                    bool(hull_include)
                    and mostrar_hull
                ),
                hull_include_set=hull_include,
                plot_key="scatter_main_3d_fixed",
            )

    # =========================================
    # Manual exclusion interface
    # =========================================

    if out_mode == "Manual":
        with st.expander(
            "Exclusión rápida por 'idx' (por cluster)"
        ):
            try:
                pca2_for_table = PCA(
                    n_components=2,
                    random_state=42,
                ).fit_transform(X_num)

            except Exception:
                pca2_for_table = np.c_[
                    np.arange(n),
                    np.zeros(n),
                ]

            df_idxview = pd.DataFrame({
                "idx": np.arange(n, dtype=int),
                "x": pca2_for_table[:, 0],
                "y": pca2_for_table[:, 1],
                "cluster": labels_all.astype(int),
            })

            cl_opts = sorted(
                np.unique(labels_all).tolist()
            )

            cl_sel = st.selectbox(
                "Cluster a visualizar",
                options=cl_opts,
                index=0,
                key="quickex_cl_sel",
            )

            sub = df_idxview[
                df_idxview["cluster"] == cl_sel
            ].copy()

            st.caption(
                "Vista: idx + coords (PCA2D)"
            )

            st.dataframe(
                sub[["idx", "x", "y"]].head(500),
                use_container_width=True,
            )

            c1, c2 = st.columns([1, 2])

            with c1:
                one_idx_text = st.text_input(
                    "Índice único a excluir",
                    value="",
                    key="quickex_one",
                )

                if st.button(
                    "Excluir índice",
                    key="quickex_one_btn",
                ):
                    try:
                        idx_val = int(
                            one_idx_text.strip()
                        )

                        exclude_point(idx_val, n)

                        st.success(
                            f"Excluido idx={idx_val}. "
                            f"Total excluidos: "
                            f"{len(st.session_state['manual_excluded_idx'])}"
                        )

                        st.rerun()

                    except Exception as e:
                        st.error(
                            f"Índice inválido: {e}"
                        )

            with c2:
                many_idx_text = st.text_input(
                    "Índices a excluir (separados por coma)",
                    value="",
                    key="quickex_many",
                )

                if st.button(
                    "Excluir lista",
                    key="quickex_many_btn",
                ):
                    try:
                        idxs = _parse_indices_text(
                            many_idx_text
                        )

                        exclude_indices(idxs, n)

                        st.success(
                            f"Excluidos {len(idxs)} índices. "
                            f"Total excluidos: "
                            f"{len(st.session_state['manual_excluded_idx'])}"
                        )

                        st.rerun()

                    except Exception as e:
                        st.error(
                            f"Entrada inválida: {e}"
                        )

            rein_txt = st.text_input(
                "Índices a reincluir (coma)",
                value="",
                key="quickex_rein",
            )

            if st.button(
                "Reincluir",
                key="quickex_rein_btn",
            ):
                try:
                    idxs = _parse_indices_text(
                        rein_txt
                    )

                    include_indices(idxs, n)

                    st.success(
                        f"Reincluidos {len(idxs)} índices. "
                        f"Total excluidos: "
                        f"{len(st.session_state['manual_excluded_idx'])}"
                    )

                    st.rerun()

                except Exception as e:
                    st.error(
                        f"Entrada inválida: {e}"
                    )

            st.info(
                "Excluidos manuales actuales "
                f"(primeros 30): "
                f"{st.session_state['manual_excluded_idx'][:30]} "
                f"| total="
                f"{len(st.session_state['manual_excluded_idx'])}"
            )

    # =========================================
    # Cluster preview
    # =========================================

    if v_show_preview:
        preview_clusters_ui(
            df=df,
            labels_all=labels_all,
            used_num=used_num,
            all_cols=all_cols,
            visible_clusters=visible_clusters,
        )

    # =========================================
    # Internal clustering metrics
    # =========================================

    if v_show_internal:
        st.subheader(
            "Métricas internas (sin ruido)"
        )

        eval_mask = labels_all >= 0

        intr = (
            cluster_internal_metrics(
                X_num[eval_mask],
                labels_all[eval_mask],
            )
            if eval_mask.any()
            else {}
        )

        st.json(intr)

    # =========================================
    # Correlation matrices
    # =========================================

    if v_show_corr_global:
        st.subheader(
            "Matriz de correlación — Global"
        )

        plot_corr(
            df[used_num],
            "Correlación (global)",
            key="corr_global_clustering",
        )

    if v_show_corr_per_cluster:
        st.subheader(
            "Matriz de correlación — por cluster"
        )

        valid_clusters = [
            c
            for c in sorted(np.unique(labels_all))
            if c >= 0
        ]

        tabs = st.tabs(
            [f"Cluster {cl}" for cl in valid_clusters]
            or ["(sin clusters)"]
        )

        if valid_clusters:
            for tab, cl in zip(
                tabs,
                valid_clusters,
            ):
                with tab:
                    # Use positional indexing to avoid
                    # mismatches with custom DataFrame indexes.
                    rows_idx = np.where(
                        labels_all == cl
                    )[0]

                    sub_df = df.iloc[rows_idx][used_num]

                    plot_corr(
                        sub_df,
                        f"Correlación — Cluster {cl}",
                        key=f"corr_cluster_{cl}",
                    )

    # =========================================
    # Cluster explanations
    # =========================================

    if v_show_explanation:
        st.subheader(
            "¿Por qué se agruparon así? (explicación)"
        )

        exp_stats = explain_clusters_numeric(
            df,
            used_num,
            labels_all,
            noise_label=-1,
        )

        st.write(
            "Top features por cluster "
            "(lift z-mean vs global):"
        )

        st.json(exp_stats)

        lr_exp = None

        try:
            mask_train_lr = labels_all >= 0

            if mask_train_lr.sum() > 1:
                lr_exp = explain_clusters_logreg(
                    X_num[mask_train_lr],
                    labels_all[mask_train_lr],
                    used_num,
                )

        except Exception:
            lr_exp = None

        if lr_exp:
            st.write(
                "Coeficientes más discriminativos "
                "por cluster (LogisticRegression):"
            )

            st.json(lr_exp)

        else:
            st.caption(
                "Coeficientes logísticos no disponibles "
                "— quizá solo un cluster válido "
                "o datos insuficientes."
            )

    # =========================================
    # Threshold analysis
    # =========================================

    if v_show_thresholds:
        with st.expander(
            "Porcentajes bajo/encima de "
            "un umbral por cluster"
        ):
            col_thr = st.selectbox(
                "Columna",
                used_num,
                key="thr_col",
            )

            thr_val_default = float(
                df[col_thr].median()
            )

            thr_val = st.number_input(
                "Umbral",
                value=thr_val_default,
                key="thr_val",
            )

            rows = []

            for cl in sorted(np.unique(labels_all)):
                if cl < 0:
                    continue

                idx = np.where(
                    labels_all == cl
                )[0]

                sub = df.iloc[idx]

                if len(sub) == 0:
                    continue

                p_bajo = float(
                    (sub[col_thr] <= thr_val).mean()
                )

                p_alto = 1.0 - p_bajo

                rows.append({
                    "cluster": int(cl),
                    f"% <= {thr_val}": round(
                        100 * p_bajo,
                        2,
                    ),
                    f"% > {thr_val}": round(
                        100 * p_alto,
                        2,
                    ),
                    "n": int(len(sub)),
                })

            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                )

    # =========================================
    # Optional regression per cluster
    # =========================================

    if v_show_supervised_block:
        with st.expander(
            "Entrenar modelo de REGRESIÓN "
            "por cluster (opcional)"
        ):
            if (
                not target_col_global
                or target_col_global not in df.columns
            ):
                st.info(
                    "Selecciona una columna objetivo "
                    "(y) numérica para entrenar "
                    "regresión por cluster."
                )

            elif not pd.api.types.is_numeric_dtype(
                df[target_col_global]
            ):
                st.warning(
                    f"La columna '{target_col_global}' "
                    "es categórica; la regresión "
                    "no aplica."
                )

            else:
                reg_name = st.selectbox(
                    "Modelo",
                    [
                        "Linear",
                        "Ridge",
                        "Lasso",
                        "KNN",
                    ],
                    index=0,
                    key="sup_reg_name",
                )

                kcv = st.number_input(
                    "Folds (KFold)",
                    min_value=2,
                    max_value=10,
                    value=5,
                    key="sup_reg_kcv",
                )

                metrics_reg = st.multiselect(
                    "Métricas",
                    [
                        "r2",
                        "rmse",
                        "mae",
                        "mape",
                    ],
                    default=[
                        "r2",
                        "rmse",
                        "mae",
                    ],
                    key="sup_reg_metrics",
                )

                if st.button(
                    "Entrenar por cluster (regresión)",
                    key="btn_sup_reg",
                ):
                    valid_clusters = [
                        int(c)
                        for c in sorted(
                            np.unique(labels_all)
                        )
                        if c >= 0
                    ]

                    for cl in valid_clusters:
                        idx = np.where(
                            labels_all == cl
                        )[0]

                        y_sub = df.iloc[idx][
                            target_col_global
                        ]

                        X_sub = df.iloc[idx][
                            used_num
                        ]

                        if len(y_sub) < kcv:
                            st.warning(
                                f"Cluster {cl}: "
                                "muestras insuficientes "
                                "para CV."
                            )
                            continue

                        pre_sub = build_preprocessor(
                            used_num,
                            scale_mode,
                            impute_strategy,
                        )

                        if reg_name == "KNN":
                            reg = KNeighborsRegressor(
                                n_neighbors=5
                            )

                        elif reg_name == "Ridge":
                            reg = Ridge(alpha=1.0)

                        elif reg_name == "Lasso":
                            reg = Lasso(alpha=0.01)

                        else:
                            reg = LinearRegression()

                        pipe = Pipeline([
                            ("pre", pre_sub),
                            ("reg", reg),
                        ])

                        cv = KFold(
                            n_splits=int(kcv),
                            shuffle=True,
                            random_state=42,
                        )

                        res = {}

                        for m in metrics_reg:
                            try:
                                if m == "r2":
                                    sc = cross_val_score(
                                        pipe,
                                        X_sub,
                                        y_sub,
                                        cv=cv,
                                        scoring="r2",
                                    )

                                elif m == "rmse":
                                    sc = -cross_val_score(
                                        pipe,
                                        X_sub,
                                        y_sub,
                                        cv=cv,
                                        scoring=(
                                            "neg_root_mean_squared_error"
                                        ),
                                    )

                                elif m == "mae":
                                    sc = -cross_val_score(
                                        pipe,
                                        X_sub,
                                        y_sub,
                                        cv=cv,
                                        scoring=(
                                            "neg_mean_absolute_error"
                                        ),
                                    )

                                elif m == "mape":
                                    vals = []

                                    for tr, te in cv.split(
                                        X_sub
                                    ):
                                        X_tr = X_sub.iloc[tr]
                                        X_te = X_sub.iloc[te]
                                        y_tr = y_sub.iloc[tr]
                                        y_te = y_sub.iloc[te]

                                        pipe.fit(
                                            X_tr,
                                            y_tr,
                                        )

                                        pred = pipe.predict(
                                            X_te
                                        )

                                        y_te_eps = np.where(
                                            np.abs(y_te) < 1e-8,
                                            1e-8,
                                            y_te,
                                        )

                                        vals.append(
                                            np.mean(
                                                np.abs(
                                                    (pred - y_te)
                                                    / y_te_eps
                                                )
                                            )
                                        )

                                    sc = np.array(vals)

                                else:
                                    continue

                                res[m] = {
                                    "mean": float(sc.mean()),
                                    "std": float(sc.std()),
                                }

                            except Exception as e:
                                res[m] = f"error: {e}"

                        st.write({
                            "cluster": int(cl),
                            **res,
                        })


# =====================================================
# REGRESSION TASK
# =====================================================

def run_regression_task(
    df: pd.DataFrame,
    used_num: List[str],
    target_col: Optional[str],
    scale_mode: str,
    impute_strategy: str,
    vis_flags: Dict,
):
    st.header("Regresión")

    if not used_num:
        st.error("Selecciona al menos una variable numérica.")
        return

    if not target_col or target_col not in df.columns:
        st.error("Selecciona una columna objetivo válida.")
        return

    if target_col in used_num:
        st.error(
            "La columna objetivo no puede estar incluida "
            "entre las variables predictoras."
        )
        return

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        st.error(
            "La regresión requiere una columna objetivo numérica."
        )
        return

    with st.sidebar:
        st.subheader("Parámetros de regresión")

        reg_name = st.selectbox(
            "Modelo de regresión",
            ["Linear", "Ridge", "Lasso", "KNN"],
            key="reg_model",
        )

        alpha = 1.0
        neighbors = 5

        if reg_name in ("Ridge", "Lasso"):
            alpha = st.number_input(
                "Regularización (alpha)",
                min_value=0.001,
                max_value=100.0,
                value=1.0,
                step=0.1,
                key="reg_alpha",
            )

        if reg_name == "KNN":
            neighbors = st.number_input(
                "Número de vecinos",
                min_value=1,
                max_value=50,
                value=5,
                key="reg_neighbors",
            )

        folds = st.slider(
            "Folds para validación cruzada",
            min_value=2,
            max_value=10,
            value=5,
            key="reg_folds",
        )

    data = df[used_num + [target_col]].copy()
    data = data.dropna(subset=[target_col])

    if len(data) < folds:
        st.error(
            "No hay suficientes observaciones para "
            "la validación cruzada."
        )
        return

    X = data[used_num]
    y = data[target_col]

    preprocessor = build_preprocessor(
        used_num,
        scale_mode,
        impute_strategy,
    )

    # Validate KNN neighbors against the smallest
    # training fold in cross-validation.
    if reg_name == "KNN":
        min_train_size = len(X) - int(
            np.ceil(len(X) / int(folds))
        )

        if int(neighbors) > min_train_size:
            st.error(
                f"KNN necesita como máximo "
                f"{min_train_size} vecinos con "
                f"{folds} folds y {len(X)} observaciones. "
                f"Actualmente seleccionaste {neighbors}. "
                "Reduce el número de vecinos."
            )
            return

    model = build_regressor(
        reg_name,
        {
            "alpha": float(alpha),
            "k": int(neighbors),
        },
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    cv = KFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=42,
    )

    if vis_flags.get("reg_cv", True):
        st.subheader("Resultados de validación cruzada")

        try:
            r2_scores = cross_val_score(
                pipe,
                X,
                y,
                cv=cv,
                scoring="r2",
            )

            rmse_scores = -cross_val_score(
                pipe,
                X,
                y,
                cv=cv,
                scoring="neg_root_mean_squared_error",
            )

            mae_scores = -cross_val_score(
                pipe,
                X,
                y,
                cv=cv,
                scoring="neg_mean_absolute_error",
            )

            st.write({
                "r2_mean": float(r2_scores.mean()),
                "r2_std": float(r2_scores.std()),
                "rmse_mean": float(rmse_scores.mean()),
                "mae_mean": float(mae_scores.mean()),
            })

        except Exception as e:
            st.error(
                f"Error en validación cruzada: {e}"
            )
            return

    try:
        pipe.fit(X, y)
        y_hat = pipe.predict(X)

    except Exception as e:
        st.error(
            f"No se pudo entrenar el modelo: {e}"
        )
        return

    st.subheader("Valores reales vs predicciones")

    results = pd.DataFrame({
        "Real": np.asarray(y),
        "Predicción": np.asarray(y_hat),
    })

    fig = px.scatter(
        results,
        x="Real",
        y="Predicción",
        title="Valores reales vs predicciones",
        opacity=0.75,
    )

    min_val = float(
        min(results["Real"].min(), results["Predicción"].min())
    )

    max_val = float(
        max(results["Real"].max(), results["Predicción"].max())
    )

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Predicción ideal",
            line=dict(dash="dash"),
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="reg_real_vs_pred",
    )

    st.caption(
        "Estas predicciones se generan sobre los datos "
        "de entrenamiento y no representan rendimiento "
        "sobre datos nuevos."
    )

    if vis_flags.get("reg_corr", True):
        st.subheader(
            "Matriz de correlación — Global (regresión)"
        )

        plot_corr(
            data[used_num + [target_col]],
            "Correlación — Global (regresión)",
            key="corr_global_reg",
        )


# =====================================================
# CLASSIFICATION TASK
# =====================================================

def run_classification_task(
    df: pd.DataFrame,
    used_num: List[str],
    target_col: Optional[str],
    scale_mode: str,
    impute_strategy: str,
    vis_flags: Dict,
):
    st.header("Clasificación")

    if not used_num:
        st.error(
            "Selecciona al menos una variable numérica."
        )
        return

    if not target_col or target_col not in df.columns:
        st.error(
            "Selecciona una columna objetivo válida."
        )
        return

    if target_col in used_num:
        st.error(
            "La columna objetivo no puede utilizarse "
            "como variable predictora."
        )
        return

    with st.sidebar:
        st.subheader("Parámetros de clasificación")

        clf_name = st.selectbox(
            "Modelo de clasificación",
            [
                "KNN",
                "LogisticRegression",
                "DecisionTree",
            ],
            key="clf_model",
        )

        neighbors = 5
        max_depth = None

        if clf_name == "KNN":
            neighbors = st.number_input(
                "Número de vecinos",
                min_value=1,
                max_value=50,
                value=5,
                key="clf_neighbors",
            )

        if clf_name == "DecisionTree":
            depth_option = st.selectbox(
                "Profundidad máxima",
                ["Sin límite", "3", "5", "10", "20"],
                key="clf_depth",
            )

            if depth_option != "Sin límite":
                max_depth = int(depth_option)

        folds = st.slider(
            "Folds para validación cruzada",
            min_value=2,
            max_value=10,
            value=5,
            key="clf_folds",
        )

    data = df[used_num + [target_col]].copy()
    data = data.dropna(subset=[target_col])

    if data.empty:
        st.error(
            "No hay observaciones válidas para clasificación."
        )
        return

    X = data[used_num]
    y = data[target_col]

    if y.nunique() < 2:
        st.error(
            "La clasificación requiere al menos dos clases."
        )
        return

    class_counts = y.value_counts()

    if len(data) < folds or class_counts.min() < folds:
        st.error(
            "Cada clase debe tener al menos tantas "
            "observaciones como folds de validación cruzada."
        )
        return

    preprocessor = build_preprocessor(
        used_num,
        scale_mode,
        impute_strategy,
    )

    # Validate KNN against the smallest training fold.
    if clf_name == "KNN":
        min_train_size = len(X) - int(
            np.ceil(len(X) / int(folds))
        )

        if int(neighbors) > min_train_size:
            st.error(
                f"KNN necesita como máximo "
                f"{min_train_size} vecinos con "
                f"{folds} folds y {len(X)} observaciones. "
                f"Actualmente seleccionaste {neighbors}. "
                "Reduce el número de vecinos."
            )
            return

    model = build_classifier(
        clf_name,
        {
            "k": int(neighbors),
            "max_depth": max_depth,
        },
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    cv = StratifiedKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=42,
    )

    if vis_flags.get("cls_cv", True):
        st.subheader(
            "Resultados de validación cruzada"
        )

        try:
            acc = cross_val_score(
                pipe,
                X,
                y,
                cv=cv,
                scoring="accuracy",
            )

            prec = cross_val_score(
                pipe,
                X,
                y,
                cv=cv,
                scoring="precision_macro",
            )

            rec = cross_val_score(
                pipe,
                X,
                y,
                cv=cv,
                scoring="recall_macro",
            )

            st.write({
                "accuracy_mean": float(acc.mean()),
                "precision_macro_mean": float(prec.mean()),
                "recall_macro_mean": float(rec.mean()),
            })

        except Exception as e:
            st.error(
                f"Error durante la validación cruzada: {e}"
            )
            return

    try:
        pipe.fit(X, y)
        y_hat = pipe.predict(X)

    except Exception as e:
        st.error(
            f"No se pudo entrenar el clasificador: {e}"
        )
        return

    if vis_flags.get("cls_cm", True):
        st.subheader(
            "Matriz de confusión "
            "(entrenado en todo el dataset — referencia)"
        )

        labels_ord = np.unique(y)

        cm = confusion_matrix(
            y,
            y_hat,
            labels=labels_ord,
        )

        cm_df = pd.DataFrame(
            cm,
            index=labels_ord,
            columns=labels_ord,
        )

        st.dataframe(
            cm_df,
            use_container_width=True,
        )

        st.caption(
            "La matriz de confusión corresponde a "
            "predicciones sobre los datos de entrenamiento."
        )

    if vis_flags.get("cls_corr", True):
        st.subheader(
            "Matriz de correlación — Global "
            "(features numéricos)"
        )

        plot_corr(
            df[used_num],
            "Correlación — Global",
            key="corr_global_clf",
        )


# =====================================================
# APP MAIN
# =====================================================

def main():
    st.set_page_config(
        page_title="SINCUP — Machine Learning Platform",
        layout="wide",
    )

    st.title(
        "SINCUP — Segmentación inteligente "
        "mediante clustering"
    )

    st.caption(
        "Plataforma de Machine Learning con clustering, "
        "clasificación, regresión, detección de outliers "
        "y visualización interactiva."
    )

    if "manual_excluded_idx" not in st.session_state:
        st.session_state["manual_excluded_idx"] = []

    with st.sidebar:
        st.header("Datos")

        file = st.file_uploader(
            "Sube 1 archivo (CSV/Excel/JSON)",
            type=[
                "csv",
                "xlsx",
                "xls",
                "xlsm",
                "json",
            ],
            key="file_up",
        )

        sample_n = st.number_input(
            "Muestrear filas (0 = sin muestreo)",
            min_value=0,
            step=100,
            value=0,
            key="sample_n",
        )

        st.header("Preprocesamiento")

        scale_mode = st.selectbox(
            "Escalado",
            [
                "Estandar (media=0, sd=1)",
                "MinMax [0,1]",
                "Robusto (mediana/IQR)",
                "Sin escalado",
            ],
            index=0,
            key="scale_mode",
        )

        impute_strategy = st.selectbox(
            "Imputación numérica",
            ["Mediana", "Media"],
            index=0,
            key="impute_mode",
        )

        st.header("Tarea")

        task = st.selectbox(
            "Selecciona",
            [
                "Clustering",
                "Clasificación",
                "Regresión",
            ],
            index=0,
            key="task_sel",
        )

        # -----------------------------------------
        # Global visibility
        # -----------------------------------------

        st.header("Visibilidad Global")

        g_show_dataset_preview = st.checkbox(
            "Vista rápida del dataset",
            True,
            key="g_show_dataset_preview",
        )

        st.caption("Clasificación")

        g_cls_cv = st.checkbox(
            "Mostrar resultados de CV (clasificación)",
            True,
            key="g_cls_cv",
        )

        g_cls_cm = st.checkbox(
            "Mostrar matriz de confusión",
            True,
            key="g_cls_cm",
        )

        g_cls_corr = st.checkbox(
            "Mostrar correlación global (clasificación)",
            True,
            key="g_cls_corr",
        )

        st.caption("Regresión")

        g_reg_cv = st.checkbox(
            "Mostrar resultados de CV (regresión)",
            True,
            key="g_reg_cv",
        )

        g_reg_corr = st.checkbox(
            "Mostrar correlación global (regresión)",
            True,
            key="g_reg_corr",
        )

    # =========================================
    # File validation
    # =========================================

    if file is None:
        st.info(
            "Sube un archivo para comenzar."
        )
        return

    try:
        df = read_any(file)

    except Exception as e:
        st.error(
            f"Error al leer archivo: {e}"
        )
        return

    if df.empty:
        st.error(
            "El archivo no contiene observaciones."
        )
        return

    # Identify the uploaded dataset by its contents.
    import hashlib

    dataset_id = hashlib.sha256(
        file.getvalue()
    ).hexdigest()

    if st.session_state.get("active_dataset_id") != dataset_id:
        st.session_state["manual_excluded_idx"] = []
        st.session_state["active_dataset_id"] = dataset_id

   
    # Reset manual exclusions when the sampling
    # configuration changes.
    current_sample_n = int(sample_n)

    if "active_sample_n" not in st.session_state:
        st.session_state["active_sample_n"] = current_sample_n

    elif st.session_state["active_sample_n"] != current_sample_n:
        st.session_state["manual_excluded_idx"] = []
        st.session_state["active_sample_n"] = current_sample_n

    # Apply reproducible sampling.
    if (
        current_sample_n > 0
        and len(df) > current_sample_n
    ):
        df = df.sample(
            n=current_sample_n,
            random_state=42,
        ).reset_index(drop=True)


    if g_show_dataset_preview:
        st.subheader(
            "Vista rápida del dataset"
        )

        st.dataframe(
            df.head(200),
            use_container_width=True,
        )

    # =========================================
    # Feature selection
    # =========================================

    all_cols = df.columns.tolist()
    num_cols = select_numeric(df)

    if not num_cols:
        st.error(
            "No hay columnas numéricas detectadas. "
            "Agrega columnas numéricas o convierte variables."
        )
        return

    with st.expander(
        "Selección de variables (opcional)"
    ):
        used_num = st.multiselect(
            "Usar columnas numéricas",
            options=num_cols,
            default=num_cols,
            key="used_num_cols",
        )

        target_col = None

        if task in ("Clasificación", "Regresión"):
            candidates = [
                c
                for c in all_cols
                if c not in used_num
            ]

            if not candidates:
                st.error(
                    "No hay columnas disponibles como objetivo. "
                    "Quita la columna objetivo de las variables "
                    "numéricas seleccionadas."
                )
                return

            target_col = st.selectbox(
                "Columna objetivo (y)",
                options=candidates,
                key="target_col_sel",
            )

    # =========================================
    # Task routing
    # =========================================

    if task == "Clustering":
        run_clustering_task(
            df=df,
            used_num=used_num,
            all_cols=all_cols,
            scale_mode=scale_mode,
            impute_strategy=impute_strategy,
            target_col_global=target_col,
        )

    elif task == "Clasificación":
        if not target_col:
            st.error(
                "Selecciona la columna objetivo (clase) "
                "en 'Selección de variables'."
            )
            return

        run_classification_task(
            df=df,
            used_num=used_num,
            target_col=target_col,
            scale_mode=scale_mode,
            impute_strategy=impute_strategy,
            vis_flags={
                "cls_cv": g_cls_cv,
                "cls_cm": g_cls_cm,
                "cls_corr": g_cls_corr,
            },
        )

    else:
        if not target_col:
            st.error(
                "Selecciona la columna objetivo "
                "para regresión."
            )
            return

        run_regression_task(
            df=df,
            used_num=used_num,
            target_col=target_col,
            scale_mode=scale_mode,
            impute_strategy=impute_strategy,
            vis_flags={
                "reg_cv": g_reg_cv,
                "reg_corr": g_reg_corr,
            },
        )

    st.divider()

    st.caption(
        "Ejecuta: streamlit run Clusters.py"
    )


if __name__ == "__main__":
    main()

