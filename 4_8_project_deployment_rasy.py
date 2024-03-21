import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.write("# Simple Iris Flower Prediction App") #klu kt gc panggil asprint
st.write("This app predicts the **SSales** Advertising!")

st.sidebar.header('User Input Parameters') 


from sklearn.linear_model import LinearRegression
modellr = LinearRegression()
modellr.fit(X_train, y_train)
y_pred = modellr.predict(X_test)

y_pred

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


print("Mean absolute error: {} ".format(mean_absolute_error(y_test, y_pred)))

print("Mean squared error: {} ".format(mean_squared_error(y_test, y_pred)))

print("Root mean squared error: {} ".format(mean_squared_error(y_test, y_pred)**0.5)) #kite x de so use this

print('Variance score: {} '.format(r2_score(y_test,y_pred)))

df_prediction = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_prediction

tempActual = pd.DataFrame()
tempActual['Actual'] = df_prediction['Actual'].copy()

tempPrediction = pd.DataFrame()
tempPrediction['Predicted'] = df_prediction['Predicted'].copy()

df_prediction['Actual'] = scalerSales.inverse_transform(tempActual)
df_prediction['Predicted'] = scalerSales.inverse_transform(tempPrediction)
df_prediction

tempActual = pd.DataFrame()
tempActual['Actual'] = df_prediction['Actual'].copy()

tempPrediction = pd.DataFrame()
tempPrediction['Predicted'] = df_prediction['Predicted'].copy()

df_prediction['Actual'] = scalerSales.inverse_transform(tempActual)
df_prediction['Predicted'] = scalerSales.inverse_transform(tempPrediction)
df_prediction

import matplotlib.pyplot as plt

df_prediction.plot(kind="bar", figsize=(30,10))

plt.title('Sales Prediction - Linear Regression')
plt.xlabel('Index')
plt.ylabel('Sales')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# The mean absolute error
print("Mean absolute error: {} ".format(mean_absolute_error(df_prediction['Actual'], df_prediction['Predicted'])))

# The mean squared error
print("Mean squared error: {} ".format(mean_squared_error(df_prediction['Actual'], df_prediction['Predicted'])))

# Root mean squared error
print("Root mean squared error: {} ".format(mean_squared_error(df_prediction['Actual'], df_prediction['Predicted'])**0.5))

# Explained variance score: 1 is perfect prediction
print('Variance score: {} '.format(r2_score(df_prediction['Actual'], df_prediction['Predicted'])))
