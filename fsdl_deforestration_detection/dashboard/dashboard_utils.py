import streamlit as st
from fastai.learner import Learner
from fastcore.dispatch import TypeDispatch

import time


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def hash_fastai_model(model):
    """Just hash the model based on its number of outputs. We don't expect
    to keep changing the model, only want to hash different outputs for
    the binary and the multilabel models."""
    return model.n_out


@st.cache(hash_funcs={Learner: hash_fastai_model})
def run_model(model, img):
    if "fastai" in str(type(model)):
        output = model.predict(img[:, :, :3])[1]
    else:
        time.sleep(1)
        output = model(img)
        output = output.round()
    return output