# Required library imports
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv('Pima_Indian_diabetes.csv')

# Adding patient Id for patient identification
data["Patient_ID"] = data.index + 1

#CHECK FOR MISSING DATA

# Converting the data types of certain features to numeric
data['Pregnancies'] = pd.to_numeric(data['Pregnancies'], errors = 'coerce')
data['BloodPressure'] = pd.to_numeric(data['BloodPressure'], errors = 'coerce')
data['SkinThickness'] = pd.to_numeric(data['SkinThickness'], errors = 'coerce')
data['BMI'] = pd.to_numeric(data['BMI'], errors = 'coerce')

# Filling 0 in place of missing values for Pregnancies
data.Pregnancies.fillna(value=0 ,inplace= True)

# Mean calculation for Glucose
Glucose_mean_1 = data[data.Outcome == 1].Glucose.mean()
Glucose_mean_0 = data[data.Outcome == 0].Glucose.mean()

# Mean calculation for BMI
BMI_mean_1 = data[data.Outcome == 1].BMI.mean()
BMI_mean_0 = data[data.Outcome == 0].BMI.mean()

# Mean calculation for skin thickness 
SkinThickness_mean_1 = data[data.Outcome == 1].SkinThickness.mean()
SkinThickness_mean_0 = data[data.Outcome == 0].SkinThickness.mean()

# Mean calculation for age
Age_mean_1 = data[data.Outcome == 1].Age.mean()
Age_mean_0 = data[data.Outcome == 0].Age.mean()

# Mean calculation for blood pressure
BloodPressure_mean_1 = data[data.Outcome == 1].BloodPressure.mean()
BloodPressure_mean_0 = data[data.Outcome == 0].BloodPressure.mean()

# Filling means in place of missing or NULL values for the rest of the features
data['Glucose'] = data.apply(lambda x: Glucose_mean_0 if np.isnan(x['Glucose']) and x['Outcome'] == 0 else Glucose_mean_1 if np.isnan(x['Glucose']) and x['Outcome'] == 1 else x['Glucose'], axis = 1)
data['SkinThickness'] = data.apply(lambda x: SkinThickness_mean_0 if np.isnan(x['SkinThickness']) and x['Outcome'] == 0 else SkinThickness_mean_1 if np.isnan(x['SkinThickness']) and x['Outcome'] == 1 else x['SkinThickness'], axis = 1)
data['BMI'] = data.apply(lambda x: BMI_mean_0 if np.isnan(x['BMI']) and x['Outcome'] == 0 else BMI_mean_1 if np.isnan(x['BMI']) and x['Outcome'] == 1 else x['BMI'], axis = 1)
data['Age'] = data.apply(lambda x: Age_mean_0 if np.isnan(x['Age']) and x['Outcome'] == 0 else Age_mean_1 if np.isnan(x['Age']) and x['Outcome'] == 1 else x['Age'], axis = 1)
data['BloodPressure'] = data.apply(lambda x: BloodPressure_mean_0 if np.isnan(x['BloodPressure']) and x['Outcome'] == 0 else BloodPressure_mean_1 if np.isnan(x['BloodPressure']) and x['Outcome'] == 1 else x['Age'], axis = 1)
data.reset_index()

# Feature vector
X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

# Target 
Y= data['Outcome']

Accuracy = []
# 1000 iterations of data splitting and logistic regression
for i in range(1000):
    # Data splitting into training and testing subset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    # Applying logistic regression
    logmodel = LogisticRegression(solver='liblinear')
    #Fitting and predicting on the training and testing data respectively
    logmodel.fit(X_train, Y_train)
    predictions = logmodel.predict(X_test)
    # Accuracy computation
    score = logmodel.score(X_test, Y_test)
    Accuracy.append(score)

# Final result
print(max(Accuracy))
print(min(Accuracy))
print(sum(Accuracy)/len(Accuracy))    

