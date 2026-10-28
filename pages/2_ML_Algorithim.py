import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import time
from utils.state_manager import init_state, next_step
from utils.data_generator import generate_cluster_data, generate_regression_data
from algorithms.ml.linear_regression import linear_regression_steps
from algorithms.ml.kmeans import kmeans_steps
from algorithms.ml.knn import knn_steps
from algorithms.ml.decision_tree import decision_tree_steps
from algorithms.ml.random_forest import random_forest_steps
from algorithms.ml.logistic_regression import logistic_regression_steps
from visualizers.ml_visualizer import (
    render_linear_regression, render_kmeans, render_knn,
    render_decision_tree, render_random_forest, render_logistic_regression
)
from utils.constants import ML_ALGORITHMS, SPEED_OPTIONS
from utils.data_loader import load_crop_yield_data, preprocess_data, get_classification_data

# Load data once
df_raw = load_crop_yield_data()
X_train, X_test, y_train, y_test, df = preprocess_data(df_raw)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = get_classification_data(df)
# ── Page Style ──────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0a0a0a; }
    .sort-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #7C3AED;
        text-align: center;
        padding: 1rem;
    }
    .step-counter {
        font-size: 1rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="sort-title">🤖 ML Algorithm Visualizer</p>', unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
algo = st.sidebar.selectbox("Choose Algorithm", ML_ALGORITHMS)
speed_label = st.sidebar.selectbox("Speed", list(SPEED_OPTIONS.keys()))
speed = SPEED_OPTIONS[speed_label]

# ── State Init ───────────────────────────────────────────────────────────────
init_state()

for key in ["test_point", "prediction", "X", "y", "rf_model"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Run Algorithm ─────────────────────────────────────────────────────────────
if st.sidebar.button(" Run Algorithm"):
    st.session_state.current_step = 0
    with st.spinner("Running algorithm..."):

        if algo == "Linear Regression":
            X, y = generate_regression_data()
            steps = linear_regression_steps(X, y)
            st.session_state.steps = steps
            st.session_state.X = X
            st.session_state.y = y

        elif algo == "KMeans":
            X, y = generate_cluster_data()
            steps = kmeans_steps(X)
            st.session_state.steps = steps
            st.session_state.X = X

        elif algo == "KNN":
            X, y = generate_cluster_data()
            test_point = X[0]
            steps, prediction = knn_steps(X, y, test_point)
            st.session_state.steps = steps
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.test_point = test_point
            st.session_state.prediction = prediction

        elif algo == "Decision Tree":
            X, y = generate_cluster_data()
            steps = decision_tree_steps(X, y)
            st.session_state.steps = steps
            st.session_state.X = X
            st.session_state.y = y

        elif algo == "Random Forest":
            X, y = generate_cluster_data()
            steps, rf_model = random_forest_steps(X, y, n_estimators=10)
            st.session_state.steps = steps
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.rf_model = rf_model

        elif algo == "Logistic Regression":
            X, y = generate_cluster_data()
            steps = logistic_regression_steps(X, y)
            st.session_state.steps = steps
            st.session_state.X = X
            st.session_state.y = y

        st.session_state.current_step = 0
        st.session_state.is_playing = False

# ── Play / Pause / Reset Controls (for animated algos) ───────────────────────
if algo in ["Linear Regression", "KMeans", "Random Forest", "Logistic Regression"]:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ Play"):
            st.session_state.is_playing = True
    with col2:
        if st.button("⏸ Pause"):
            st.session_state.is_playing = False
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.current_step = 0
            st.session_state.is_playing = False

# ── Render ────────────────────────────────────────────────────────────────────
if len(st.session_state.steps) > 0:

    if algo == "Linear Regression":
        st.markdown(f'<p class="step-counter">Step {st.session_state.current_step + 1} of {len(st.session_state.steps)}</p>', unsafe_allow_html=True)
        render_linear_regression(
            st.session_state.X,
            st.session_state.y,
            st.session_state.steps,
            st.session_state.current_step
        )
        if st.session_state.is_playing:
            time.sleep(speed)
            next_step()
            st.rerun()

    elif algo == "KMeans":
        st.markdown(f'<p class="step-counter">Step {st.session_state.current_step + 1} of {len(st.session_state.steps)}</p>', unsafe_allow_html=True)
        if st.session_state.steps and st.session_state.current_step < len(st.session_state.steps):
            render_kmeans(
        st.session_state.X,
        st.session_state.steps,
        st.session_state.current_step
                )
        if st.session_state.is_playing:
            time.sleep(speed)
            next_step()
            st.rerun()

    elif algo == "KNN":
        if st.session_state.test_point is not None:
            render_knn(
                st.session_state.X,
                st.session_state.y,
                st.session_state.steps,
                st.session_state.test_point
            )
            st.markdown(f"""
                <div style='background:#1a1a1a; border:1px solid #7C3AED; border-radius:10px;
                            padding:1rem; text-align:center; color:white; margin-top:1rem;'>
                    <h3 style='color:#7C3AED;'>Predicted Class: <span style='color:#B7FC67;'>{st.session_state.prediction}</span></h3>
                </div>
            """, unsafe_allow_html=True)

    elif algo == "Decision Tree":
        render_decision_tree(
            st.session_state.X,
            st.session_state.y,
            st.session_state.steps
        )

    elif algo == "Random Forest":
        st.markdown(f'<p class="step-counter">Trees: {st.session_state.current_step + 1} of {len(st.session_state.steps)}</p>', unsafe_allow_html=True)
        render_random_forest(
            st.session_state.steps,
            st.session_state.current_step
        )
        if st.session_state.is_playing:
            time.sleep(speed)
            next_step()
            st.rerun()

    elif algo == "Logistic Regression":
        st.markdown(f'<p class="step-counter">Iteration {st.session_state.current_step + 1} of {len(st.session_state.steps)}</p>', unsafe_allow_html=True)
        render_logistic_regression(
            st.session_state.X,
            st.session_state.y,
            st.session_state.steps,
            st.session_state.current_step
        )
        if st.session_state.is_playing:
            time.sleep(speed)
            next_step()
            st.rerun()

else:
    st.info(" Click '▶ Run Algorithm' in the sidebar to start!")