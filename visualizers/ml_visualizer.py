import streamlit as st
import plotly.graph_objects as go
import numpy as np
from visualizers.chart_helpers import get_default_layout


def render_linear_regression(X, y, steps, current_step):
    m, b = steps[current_step]
    y_line = m * X + b

    fig = go.Figure(data=[
        go.Scatter(x=X.flatten(), y=y, mode='markers',
                   marker=dict(color='#00FFFF'), name='Data Points'),
        go.Scatter(x=X.flatten(), y=y_line.flatten(), mode='lines',
                   line=dict(color='#B7FC67', width=3), name='Regression Line')
    ], layout=get_default_layout(f"Linear Regression — Step {current_step + 1} of {len(steps)}"))

    st.plotly_chart(fig, use_container_width=True)


def render_kmeans(X, steps, current_step):
    current_step = min(current_step, len(steps) - 1)
    centroids, labels = steps[current_step]

    fig = go.Figure(data=[
        go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode='markers',
            marker=dict(color=labels.tolist(), colorscale='Viridis', size=8),
            name='Data Points'
        ),
        go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1],
            mode='markers',
            marker=dict(size=18, color='red', symbol='x', line=dict(width=3)),
            name='Centroids'
        )
    ], layout=get_default_layout(f"KMeans Clustering — Step {current_step + 1} of {len(steps)}"))

    st.plotly_chart(fig, use_container_width=True)


def render_knn(X, y, steps, test_point):
    k_indices, k_labels, distances = steps[0]

    fig = go.Figure(data=[
        go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode='markers',
            marker=dict(color=y.tolist(), colorscale='Viridis', size=8),
            name='Data Points'
        ),
        go.Scatter(
            x=X[k_indices, 0], y=X[k_indices, 1],
            mode='markers',
            marker=dict(size=16, color='yellow', symbol='circle-open', line=dict(width=3)),
            name='K Neighbours'
        ),
        go.Scatter(
            x=[test_point[0]], y=[test_point[1]],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star'),
            name='Test Point'
        )
    ], layout=get_default_layout("KNN Classification"))

    st.plotly_chart(fig, use_container_width=True)


def render_decision_tree(X, y, steps):
    model, predictions, importances = steps[0]

    accuracy = np.mean(predictions == y) * 100

    col1, col2 = st.columns(2)

    with col1:
        # Feature importance bar chart
        feature_names = [f"Feature {i+1}" for i in range(len(importances))]
        fig = go.Figure(data=[
            go.Bar(
                x=feature_names,
                y=importances,
                marker_color='#7C3AED'
            )
        ], layout=get_default_layout("Feature Importances"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Decision boundary scatter
        fig2 = go.Figure(data=[
            go.Scatter(
                x=X[:, 0], y=X[:, 1],
                mode='markers',
                marker=dict(
                    color=predictions.tolist(),
                    colorscale='Viridis',
                    size=8,
                    showscale=True
                ),
                name='Predictions'
            )
        ], layout=get_default_layout("Decision Boundary"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
        <div style='background:#1a1a1a; border:1px solid #7C3AED; border-radius:10px;
                    padding:1rem; text-align:center; color:white; margin-top:1rem;'>
            <h3 style='color:#7C3AED;'>Model Accuracy</h3>
            <h2>{accuracy:.2f}%</h2>
        </div>
    """, unsafe_allow_html=True)


def render_random_forest(steps, current_step):
    n_trees, accuracy, importances = steps[current_step]

    col1, col2 = st.columns(2)

    with col1:
        # Accuracy vs number of trees
        tree_counts = [s[0] for s in steps[:current_step + 1]]
        accuracies = [s[1] * 100 for s in steps[:current_step + 1]]

        fig = go.Figure(data=[
            go.Scatter(
                x=tree_counts,
                y=accuracies,
                mode='lines+markers',
                line=dict(color='#7C3AED', width=3),
                marker=dict(color='#B7FC67', size=8)
            )
        ], layout=get_default_layout(f"Accuracy vs Trees — {n_trees} trees"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature importances
        feature_names = [f"Feature {i+1}" for i in range(len(importances))]
        fig2 = go.Figure(data=[
            go.Bar(x=feature_names, y=importances, marker_color='#00FFFF')
        ], layout=get_default_layout("Feature Importances"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
        <div style='background:#1a1a1a; border:1px solid #7C3AED; border-radius:10px;
                    padding:1rem; text-align:center; color:white; margin-top:1rem;'>
            <h3 style='color:#7C3AED;'>Trees: {n_trees} &nbsp;|&nbsp; Accuracy: {accuracy*100:.2f}%</h3>
        </div>
    """, unsafe_allow_html=True)


def render_logistic_regression(X, y, steps, current_step):
    predictions, accuracy, probabilities, model = steps[current_step]

    col1, col2 = st.columns(2)

    with col1:
        # Scatter of predictions
        fig = go.Figure(data=[
            go.Scatter(
                x=X[:, 0], y=X[:, 1],
                mode='markers',
                marker=dict(
                    color=predictions.tolist(),
                    colorscale='RdYlGn',
                    size=8,
                    showscale=True
                ),
                name='Predictions'
            )
        ], layout=get_default_layout("Decision Boundary"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Accuracy over iterations
        accuracies = [s[1] * 100 for s in steps[:current_step + 1]]
        fig2 = go.Figure(data=[
            go.Scatter(
                x=list(range(1, len(accuracies) + 1)),
                y=accuracies,
                mode='lines+markers',
                line=dict(color='#B7FC67', width=3),
                marker=dict(color='#7C3AED', size=8)
            )
        ], layout=get_default_layout("Accuracy Over Iterations"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
        <div style='background:#1a1a1a; border:1px solid #7C3AED; border-radius:10px;
                    padding:1rem; text-align:center; color:white; margin-top:1rem;'>
            <h3 style='color:#7C3AED;'>Iteration {current_step + 1} of {len(steps)}</h3>
            <h2>Accuracy: {accuracy*100:.2f}%</h2>
        </div>
    """, unsafe_allow_html=True)