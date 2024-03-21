import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import pickle

st.write("# Simple Advertising Prediction App")
st.write("This app predicts the **Sales** Advertising!")

st.sidebar.header('User Input Parameters') 

def user_input_features():
    TV = st.sidebar.slider('TV', 1.0, 5.0, 10.0) 
    Radio = st.sidebar.slider('Radio', 1.0, 5.0, 10.0)
    Newspaper = st.sidebar.slider('Newspaper', 1.0, 5.0, 10.0)
    
    data = {'TV': TV, 
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])  
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)


loaded_model = pickle.load(open("Advertising.h5", "rb")) #rb: read binary
prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)


  
     
