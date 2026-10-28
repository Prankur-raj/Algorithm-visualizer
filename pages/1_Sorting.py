import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import time


from utils.state_manager import init_state, reset_state, next_step
from utils.data_generator import generate_random_array
from algorithms.sorting.bubble_sort import bubble_sort_steps
from algorithms.sorting.merge_sort import merge_sort_steps
from algorithms.sorting.quick_sort import  quick_sort_steps
from visualizers.sort_visualizer import render_sort
from utils.constants import SORTING_ALGORITHMS, SPEED_OPTIONS
from config.settings import DEFAULT_ARRAY_SIZE, MIN_ARRAY_SIZE, MAX_ARRAY_SIZE


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
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="sort-title">⚡ Sorting Visualizer</p>', unsafe_allow_html=True)

algo = st.sidebar.selectbox("Choose Algorithm", SORTING_ALGORITHMS)
size = st.sidebar.slider("Array Size", MIN_ARRAY_SIZE, MAX_ARRAY_SIZE, DEFAULT_ARRAY_SIZE)
speed_label = st.sidebar.selectbox("Speed", list(SPEED_OPTIONS.keys()))
speed = SPEED_OPTIONS[speed_label]


init_state()

if st.sidebar.button("Generate New Array"):
    arr = generate_random_array(size)
    
    if algo == "Bubble Sort":
        steps = bubble_sort_steps(arr)
    elif algo == "Merge Sort":
        steps = merge_sort_steps(arr)
    elif algo == "Quick Sort":
        steps = quick_sort_steps(arr)
    
    st.session_state.steps = steps
    reset_state()

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶ Play"):
        st.session_state.is_playing = True

with col2:
    if st.button("⏸ Pause"):
        st.session_state.is_playing = False

with col3:
    if st.button("🔄 Reset"):
        reset_state()


if len(st.session_state.steps) > 0:
    st.markdown(f'<p class="step-counter">Step {st.session_state.current_step + 1} of {len(st.session_state.steps)}</p>', unsafe_allow_html=True)
    render_sort(st.session_state.steps, st.session_state.current_step)
    if st.session_state.is_playing:
        time.sleep(speed)
        next_step()
        st.rerun()
else:
    st.info("Click 'Generate New Array' in the sidebar to start!")