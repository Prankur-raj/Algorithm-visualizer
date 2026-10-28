import sys
import os 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from config.settings import APP_TITLE, PAGE_ICON, PAGE_LAYOUT



st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout=PAGE_LAYOUT
)

st.markdown("""
    <style>
    .main { background-color: #0a0a0a; }
    .welcome-title {
        font-size: 3rem;
        font-weight: 800;
        color: #7C3AED;
        text-align: center;
        padding: 2rem;
        animation: fadeIn 2s ease-in;
    }
    .welcome-sub {
        font-size: 1.2rem;
        color: #ffffff;
        text-align: center;
        padding: 0.5rem;
    }
    .card {
        background-color: #1a1a1a;
        border: 1px solid #7C3AED;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        color: #ffffff;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="welcome-title">⚡ Algorithm Visualizer</p>', unsafe_allow_html=True)
st.markdown('<p class="welcome-sub">Learn algorithms through real-time interactive visualizations</p>', unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="card">
            <h2>🔢 Sorting Algorithms</h2>
            <p>Visualize Bubble Sort, Merge Sort and Quick Sort step by step</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="card">
            <h2>🤖 ML Algorithms</h2>
            <p>Watch KMeans, Linear Regression and KNN learn from data</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p class="welcome-sub">👈 Pick a page from the sidebar to get started!</p>', unsafe_allow_html=True)

