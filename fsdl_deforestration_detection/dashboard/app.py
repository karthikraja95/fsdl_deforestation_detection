from fsdl_deforestration_detection.dashboard.layouts import overview, playground
import streamlit as st
import layouts

PAGES = dict(Playground=layouts.playground, Overview=layouts.overview)

st.sidebar.title("About")
st.sidebar.info(
    "Full Stack Deep Learning course project on detecting deforestation through satellite imagery."
)
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page()