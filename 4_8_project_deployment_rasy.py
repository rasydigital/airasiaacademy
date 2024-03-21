import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
st.write("#Advertising Sale Prediction")
st.write("This app predicts the **Sales** Advertising!")

st.sidebar.header('User Input Parameters') 

def user_input_features():
    TV = st.sidebar.slider('TV', 4.3, 7.9, 5.4) 
    Radio = st.sidebar.slider('Radio', 2.0, 4.4, 3.4)
    Newspaper = st.sidebar.slider('Newspaper', 1.0, 6.9, 1.3)
    
    data = {'TV': TV, 
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])  
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)



prediction = modellr.predict(df) 

modelrr = LinearRegression()
modelrr.fit(X, Y)

st.subheader('Prediction')
st.write(prediction)


  
     
