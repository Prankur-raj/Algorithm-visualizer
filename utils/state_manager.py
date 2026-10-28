import streamlit as st

def init_state():

    if "current_step" not in st.session_state:
        st.session_state.current_step=0
    if "is_playing" not in st.session_state:
         st.session_state.is_playing=False 
    if "speed" not in st.session_state:
        st.session_state.speed=0.1
    if "steps" not in st.session_state:  
         st.session_state.steps=[]  


def reset_state():
     st.session_state.current_step=0

     st.session_state.is_playing=False

def next_step():

    if  st.session_state.current_step<len(st.session_state.steps)-1:
         st.session_state.current_step+=1
    else:
         st.session_state.is_playing=False