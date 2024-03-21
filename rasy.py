import streamlit as st
import numpy as np
import pandas as pd

st.header("Advertising Sale")
option = st.sidebar.selectbox(
    'Select Option',
     ['line chart','map','T n C','Long Process'])
