import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import LinearRegression
st.write("# Advertising Sale Prediction")
st.write("This app predicts the **Sales** Advertising!")

st.sidebar.header('User Input Parameters') 

def user_input_features():
    TV = st.sidebar.slider('TV', 4.3, 7.9, 5.4) #slider min, max ,def
    Radio = st.sidebar.slider('Radio', 2.0, 4.4, 3.4)
    Newspaper = st.sidebar.slider('Newspaper', 1.0, 6.9, 1.3)
    
    data = {'TV': TV, 
            'Radio': Radio,
            'Newspaper': Newspaper,
            
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = sns.load_dataset('Advertising')
X = data.drop(['sales'],axis=1)
Y = data.Sales.copy() 

modellr = LinearRegression()
modellr.fit(X, y)

prediction = modellr.predict(df) 
prediction_proba = modellr.predict_proba(df) 

st.subheader('Class labels and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

  
     
