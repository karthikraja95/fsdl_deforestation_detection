import streamlit as st

import time


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache
def run_model(model, img):
    time.sleep(1)
    output = model(img)
    output = output.round()
    return output