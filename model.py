from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the Data Set
rain_data = pd.read_csv('Harare_Weather.csv')

# Feature Engineering of Date column to decrease high cardinality:
rain_data['Date time'] = pd.to_datetime(rain_data['Date time'])
YEAR = []
MONTH = []
DAY = []
for i in range(len(rain_data)):
    DAY.append(rain_data['Date time'][i].day)
    MONTH.append(rain_data['Date time'][i].month)
    YEAR.append(rain_data['Date time'][i].year)

rain_data['Year'] = YEAR
rain_data['Month'] = MONTH
rain_data['Day'] = DAY 

# Drop Date column
rain_data.drop(columns = {'Name','Date time','Conditions'},axis = 1, inplace = True)
rain_data.head()

#Dropping the columns
rain_data.drop(columns = {'Snow', 'Snow Depth','Wind Chill','Heat Index','Wind Gust'}, inplace = True)

#Reloading the new numerical features in the dataset
num_features = [col_name for col_name in rain_data.columns if rain_data[col_name].dtype != 'O']
print("Number of Numerical Features: {}".format(len(num_features)))
print("Numerical Features: ",num_features)

# Imputing missing values in numerical features using mean.
num_features_with_null = [feature for feature in num_features if rain_data[feature].isnull().sum()]
for feature in num_features_with_null:
    mean_value = rain_data[feature].mean()
    rain_data[feature].fillna(mean_value,inplace=True)

# Encoding of Categorical Features
rain_data['Raining Today'].replace({'No':0, 'Yes': 1}, inplace = True)
rain_data['Raining Tomorrow'].replace({'No':0, 'Yes': 1}, inplace = True)

# Splitting data into Independent Features and Dependent Features:
X = rain_data.drop(['Raining Tomorrow'],axis=1)
y = rain_data['Raining Tomorrow']

#Splitting Data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Feature Scaling
scaler=MinMaxScaler(feature_range=(0,1))
X_train=scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training using Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier_logreg = LogisticRegression(solver='saga', penalty = 'l1',C=1.0,random_state=0)
classifier_logreg.fit(X_train, y_train)

# Saving the Model and Scaling object with Pickle
import pickle
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file) # scaler is an object of MinMaxScaler class.

with open('logreg.pkl', 'wb') as file:
    pickle.dump(classifier_logreg, file) # here classifier_logreg is trained model