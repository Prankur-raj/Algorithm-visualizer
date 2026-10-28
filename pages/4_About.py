import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st

st.markdown("""
    <style>
    .sort-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #7C3AED;
        text-align: center;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="sort-title">📖 About</p>', unsafe_allow_html=True)

st.markdown("""Algorithm Visualizer is an interactive web application designed to simplify complex algorithms through intuitive and real-time visualizations.

The goal of this app is to help students and beginners understand algorithms not just through theory, but by seeing how they work step by step.

🔹 What You Can Do
🔢 Visualize different sorting algorithms (Bubble Sort, Merge Sort, Quick Sort, etc.)
⚡ Control the speed of execution
🎯 Observe step-by-step comparisons and swaps
📊 Generate custom or random datasets
🤖 Explore basic Machine Learning algorithm visualizations
🔹 Why This App?

Understanding algorithms from code alone can be difficult. This app makes learning:

Visual – See how algorithms behave
Interactive – Control inputs and experiment
Intuitive – Grasp concepts faster and more clearly
🔹 How It Works

The user selects an algorithm, sets parameters like array size and speed, and the app runs the algorithm in real time. Each step is visualized graphically, highlighting important operations such as comparisons and swaps.

🔹 Built With
Python
Streamlit
NumPy, Matplotlib / Plotly
Scikit-learn
🔹 Developer Note

This project was built as a learning tool to make Data Structures and Machine Learning concepts easier to understand through visualization.

🔥 Short Version

Algorithm Visualizer is a web-based tool that helps users understand sorting and machine learning algorithms through real-time interactive visualizations.
            
   
""")


st.markdown("### Algorithms")
st.table({
    "Algorithm": ["Bubble Sort", "Merge Sort", "Quick Sort", "KMeans", "Linear Regression", "KNN"],
   "Type": ["Sorting", "Sorting", "Sorting", "ML - Clustering", "ML - Regression", "ML - Classification"],
"Time Complexity": ["O(n²)", "O(n log n)", "O(n log n)", "O(n×k×i)", "O(n)", "O(n)"]
})


st.markdown("""
    ### 🚀 How to Use

    **For Sorting Algorithms:**
    1. Go to **Sorting Visualizer** page from sidebar
    2. Choose an algorithm from dropdown
    3. Set your array size using the slider
    4. Choose animation speed
    5. Click **Generate New Array**
    6. Press **▶ Play** to start animation
    7. Use **⏸ Pause** to stop at any step
    8. Use **🔄 Reset** to go back to start

    **For ML Algorithms:**
    1. Go to **ML Algorithm Visualizer** page from sidebar
    2. Choose an algorithm from dropdown
    3. Click **Run Algorithm**
    4. Watch the algorithm learn step by step
""")