import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dataset Overview", layout="wide")

# ── Styling ─────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #7C3AED;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .section-title {
        color: #A78BFA;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Dataset Overview & ML Analysis")
st.markdown("Upload your agricultural dataset to explore it and run ML algorithms on it.")

# ── Upload Section ───────────────────────────────────────────
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # ── Tabs ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Data Preview",
        " Statistics",
        " Visualizations",
        " Correlation",
        "ML Analysis"
    ])

    DARK_LAYOUT = dict(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font_color='white',
        xaxis=dict(gridcolor='#2a2a3e'),
        yaxis=dict(gridcolor='#2a2a3e')
    )

    # ── Tab 1: Data Preview ───────────────────────────────────
    with tab1:
        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head(50), use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", df_raw.shape[0])
        col2.metric("Total Columns", df_raw.shape[1])
        col3.metric("Missing Values", int(df_raw.isnull().sum().sum()))
        col4.metric("Duplicate Rows", int(df_raw.duplicated().sum()))

        st.subheader("Column Info")
        info_df = pd.DataFrame({
            'Column': df_raw.columns,
            'Data Type': df_raw.dtypes.values,
            'Missing Values': df_raw.isnull().sum().values,
            'Unique Values': df_raw.nunique().values,
            'Sample Value': [df_raw[c].dropna().iloc[0] if len(df_raw[c].dropna()) > 0 else 'N/A' for c in df_raw.columns]
        })
        st.dataframe(info_df, use_container_width=True)

    # ── Tab 2: Statistics ─────────────────────────────────────
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(df_raw.describe(), use_container_width=True)

        st.subheader("Missing Values")
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            fig = px.bar(
                x=missing.index, y=missing.values,
                labels={'x': 'Column', 'y': 'Missing Count'},
                title="Missing Values per Column",
                color=missing.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")

    # ── Tab 3: Visualizations ─────────────────────────────────
    with tab3:
        numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df_raw.select_dtypes(include='object').columns.tolist()

        # Distribution
        st.subheader(" Distribution Plot")
        if numeric_cols:
            selected_col = st.selectbox("Select numeric column", numeric_cols, key='dist_col')
            fig = px.histogram(
                df_raw, x=selected_col, nbins=50,
                title=f"Distribution of {selected_col}",
                color_discrete_sequence=['#7C3AED']
            )
            fig.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # Bar Chart
        st.subheader(" Bar Chart")
        if categorical_cols and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                cat_col = st.selectbox("Category column", categorical_cols, key='bar_cat')
            with col2:
                num_col = st.selectbox("Value column", numeric_cols, key='bar_num')

            top_n = st.slider("Top N categories", 5, 30, 10)
            agg_df = df_raw.groupby(cat_col)[num_col].mean().nlargest(top_n).reset_index()
            fig = px.bar(
                agg_df, x=cat_col, y=num_col,
                title=f"Average {num_col} by {cat_col} (Top {top_n})",
                color=num_col, color_continuous_scale='Viridis'
            )
            fig.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter
        st.subheader(" Scatter Plot")
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X axis", numeric_cols, index=0, key='scatter_x')
            with col2:
                y_col = st.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols)-1), key='scatter_y')
            color_col = st.selectbox("Color by", ['None'] + categorical_cols, key='scatter_color')
            fig = px.scatter(
                df_raw, x=x_col, y=y_col,
                color=None if color_col == 'None' else color_col,
                title=f"{x_col} vs {y_col}", opacity=0.6
            )
            fig.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # Line Chart
        st.subheader(" Line Chart (Trend over time)")
        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                x_line = st.selectbox("X axis (time/index)", numeric_cols, key='line_x')
            with col2:
                y_line = st.selectbox("Y axis (value)", numeric_cols, key='line_y')
            group_col = st.selectbox("Group by (optional)", ['None'] + categorical_cols, key='line_group')

            if group_col != 'None':
                top_cats = df_raw[group_col].value_counts().nlargest(5).index
                line_df = df_raw[df_raw[group_col].isin(top_cats)]
                agg_line = line_df.groupby([x_line, group_col])[y_line].mean().reset_index()
                fig = px.line(agg_line, x=x_line, y=y_line, color=group_col,
                              title=f"{y_line} over {x_line} grouped by {group_col}")
            else:
                agg_line = df_raw.groupby(x_line)[y_line].mean().reset_index(name=f"{y_line}_mean")
                fig = px.line(agg_line, x=x_line, y=y_line,
                              title=f"Average {y_line} over {x_line}",
                              color_discrete_sequence=['#7C3AED'])
            fig.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Correlation ────────────────────────────────────
    with tab4:
        st.subheader(" Correlation Heatmap")
        numeric_df = df_raw.select_dtypes(include=np.number)
        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr()
            fig = px.imshow(
                corr, text_auto=True,
                color_continuous_scale='RdBu',
                title="Feature Correlation Heatmap",
                aspect="auto"
            )
            fig.update_layout(paper_bgcolor='#0e1117', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for correlation heatmap.")

    # ── Tab 5: ML Analysis ────────────────────────────────────
    with tab5:
        st.subheader("ML Analysis on Your Dataset")
        st.markdown("Select features, target column, and algorithm — then run the model!")

        all_cols = df_raw.columns.tolist()
        numeric_cols_ml = df_raw.select_dtypes(include=np.number).columns.tolist()

        # ── Config ────────────────────────────────────────────
        st.markdown("### ⚙️ Configuration")
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox(" Target Column (what to predict)", all_cols)
        with col2:
            algorithm = st.selectbox(" Algorithm", [
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "K-Nearest Neighbors",
                "Logistic Regression"
            ])

        feature_cols = st.multiselect(
            " Feature Columns (inputs)",
            [c for c in all_cols if c != target_col],
            default=[c for c in numeric_cols_ml if c != target_col][:3]
        )

        # Hyperparameters
        st.markdown("###  Hyperparameters")
        hp_col1, hp_col2, hp_col3 = st.columns(3)
        with hp_col1:
            test_size = st.slider("Test Size %", 10, 40, 20) / 100
        with hp_col2:
            if algorithm == "K-Nearest Neighbors":
                k_val = st.slider("K (neighbors)", 1, 20, 5)
            elif algorithm in ["Decision Tree", "Random Forest"]:
                max_depth = st.slider("Max Depth", 1, 20, 5)
            elif algorithm == "Random Forest":
                n_estimators = st.slider("N Estimators", 10, 200, 100)
            else:
                st.empty()
        with hp_col3:
            if algorithm == "Random Forest":
                n_estimators = st.slider("N Estimators", 10, 200, 100)
            else:
                st.empty()

        # ── Run Button ────────────────────────────────────────
        if st.button(" Run ML Algorithm", use_container_width=True):
            if not feature_cols:
                st.error("Please select at least one feature column!")
            else:
                try:
                    # Prepare data
                    df_ml = df_raw[feature_cols + [target_col]].dropna()

                    # Encode categorical features
                    X = df_ml[feature_cols].copy()
                    y = df_ml[target_col].copy()

                    for col in X.select_dtypes(include='object').columns:
                        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

                    if y.dtype == 'object':
                        y = LabelEncoder().fit_transform(y.astype(str))
                        task_type = 'classification'
                    else:
                        # Auto-detect: if few unique values → classification
                        if y.nunique() <= 10:
                            task_type = 'classification'
                        else:
                            task_type = 'regression' if algorithm == 'Linear Regression' else 'classification'

                    # Force regression for Linear Regression
                    if algorithm == "Linear Regression":
                        task_type = 'regression'

                    # Scale
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=42
                    )

                    # Train model
                    if algorithm == "Linear Regression":
                        model = LinearRegression()
                    elif algorithm == "Decision Tree":
                        model = DecisionTreeClassifier(max_depth=max_depth) if task_type == 'classification' else DecisionTreeRegressor(max_depth=max_depth)
                    elif algorithm == "Random Forest":
                        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth) if task_type == 'classification' else RandomForestRegressor(n_estimators=n_estimators)
                    elif algorithm == "K-Nearest Neighbors":
                        model = KNeighborsClassifier(n_neighbors=k_val)
                        task_type = 'classification'
                    elif algorithm == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000, solver='lbfgs')
                        task_type = 'classification'

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.success(f"✅ {algorithm} trained successfully on {len(X_train)} samples!")

                    # ── Metrics ───────────────────────────────
                    st.markdown("### 📊 Performance Metrics")

                    if task_type == 'regression':
                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)

                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("MAE", f"{mae:.2f}")
                        mc2.metric("MSE", f"{mse:.2f}")
                        mc3.metric("RMSE", f"{rmse:.2f}")
                        mc4.metric("R² Score", f"{r2:.4f}")

                        # Actual vs Predicted
                        st.markdown("### 📈 Actual vs Predicted")
                        fig = px.scatter(
                            x=y_test[:200], y=y_pred[:200],
                            labels={'x': 'Actual', 'y': 'Predicted'},
                            title="Actual vs Predicted Values",
                            color_discrete_sequence=['#7C3AED']
                        )
                        min_val = min(y_test.min(), y_pred.min())
                        max_val = max(y_test.max(), y_pred.max())
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val], y=[min_val, max_val],
                            mode='lines', name='Perfect Fit',
                            line=dict(color='#10B981', dash='dash')
                        ))
                        fig.update_layout(**DARK_LAYOUT)
                        st.plotly_chart(fig, use_container_width=True)

                        # Residuals
                        st.markdown("### 📉 Residuals Plot")
                        residuals = y_test - y_pred
                        fig2 = px.histogram(
                            x=residuals, nbins=50,
                            title="Residuals Distribution",
                            color_discrete_sequence=['#F59E0B']
                        )
                        fig2.update_layout(**DARK_LAYOUT)
                        st.plotly_chart(fig2, use_container_width=True)

                    else:
                        acc = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted')

                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Accuracy", f"{acc*100:.2f}%")
                        mc2.metric("F1 Score", f"{f1:.4f}")
                        mc3.metric("Test Samples", len(y_test))

                        # Confusion Matrix
                        st.markdown("### 🔲 Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig = px.imshow(
                            cm, text_auto=True,
                            color_continuous_scale='Purples',
                            title="Confusion Matrix",
                            labels=dict(x="Predicted", y="Actual")
                        )
                        fig.update_layout(paper_bgcolor='#0e1117', font_color='white')
                        st.plotly_chart(fig, use_container_width=True)

                        # Prediction distribution
                        st.markdown("### 📊 Prediction Distribution")
                        pred_counts = pd.Series(y_pred).value_counts().reset_index()
                        pred_counts.columns = ['Class', 'Count']
                        fig2 = px.bar(
                            pred_counts, x='Class', y='Count',
                            title="Predicted Class Distribution",
                            color='Count', color_continuous_scale='Viridis'
                        )
                        fig2.update_layout(**DARK_LAYOUT)
                        st.plotly_chart(fig2, use_container_width=True)

                    # ── Feature Importance ────────────────────
                    if algorithm in ["Decision Tree", "Random Forest"]:
                        st.markdown("### 🏆 Feature Importance")
                        importances = model.feature_importances_
                        fi_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': importances
                        }).sort_values('Importance', ascending=True)

                        fig = px.bar(
                            fi_df, x='Importance', y='Feature',
                            orientation='h',
                            title="Feature Importance",
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(**DARK_LAYOUT)
                        st.plotly_chart(fig, use_container_width=True)

                    elif algorithm == "Linear Regression":
                        st.markdown("### 🏆 Feature Coefficients")
                        coef_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Coefficient': model.coef_
                        }).sort_values('Coefficient', ascending=True)

                        fig = px.bar(
                            coef_df, x='Coefficient', y='Feature',
                            orientation='h',
                            title="Feature Coefficients",
                            color='Coefficient',
                            color_continuous_scale='RdBu'
                        )
                        fig.update_layout(**DARK_LAYOUT)
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Tip: Make sure your feature columns are numeric or can be encoded, and your target column is valid.")

else:
    st.info(" Upload a CSV file above to get started.")
    st.markdown("""
    ### What you can explore:
    -  **Data Preview** — view raw data and column info
    -  **Statistics** — summary stats and missing values
    -  **Visualizations** — distributions, bar charts, scatter plots, line charts
    -  **Correlation** — heatmap of feature relationships
    -  **ML Analysis** — run any ML algorithm on your data and see results!

    ### Recommended datasets:
    - `yield.csv` — Crop yield by country and year
    - `rainfall.csv` — Annual rainfall data
    - `temp.csv` — Temperature data
    - `pesticides.csv` — Pesticide usage data
    """)