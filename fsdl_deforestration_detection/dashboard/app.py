import streamlit as st
from dashboard_utils import load_css
import layouts

PAGES = dict(Playground=layouts.playground, Overview=layouts.overview)
load_css("style.css")

st.sidebar.title("About")
st.sidebar.info(
    "Full Stack Deep Learning course project on detecting deforestation through satellite imagery."
)
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page()