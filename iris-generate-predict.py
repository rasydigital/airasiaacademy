#Utk generTE python filr kite gune this utk geennrare #should have pyhton code .py as in GC use np 
#h5. create model
#kt sini x perlu nk evluate sume - skip
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.write("# Simple Iris Flower Prediction App") #klu kt gc panggil asprint
st.write("This app predicts the **Iris flower** type!")

st.sidebar.header('User Input Parameters') #keluar sidebare kt tp

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4) #slider min, max ,def
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length, #key:values - Sepal_length - is the keyword
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features() #function ni adelah formula yg atas tu only.Lepas kt bwh dah jd 1 df

st.subheader('User Input parameters')
st.write(df)

data = sns.load_dataset('iris')
X = data.drop(['species'],axis=1) #x feature - so buang species
Y = data.species.copy() #y adalah valuable

modelGaussianIris = GaussianNB()
modelGaussianIris.fit(X, Y)

prediction = modelGaussianIris.predict(df) #thens save it on this variable
prediction_proba = modelGaussianIris.predict_proba(df) #this one exceptional

st.subheader('Class labels and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
