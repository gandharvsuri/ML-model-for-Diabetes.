import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


#Load data
data = pd.read_csv('Pima_Indian_diabetes.csv')


#Adding patient Id for patient identification
data["Patient_ID"] = data.index + 1




#CHECK FOR MISSING DATA
#Checking number of NULL Values in each column
#Comment following each print statement is the number of missing values of the corresponding column.
def Number_of_missing_values():
    print(data.Pregnancies.isnull().sum()) 
    #26
    print(data.Glucose.isnull().sum())
    #16 
    print(data.BloodPressure.isnull().sum()) 
    #0 
    print(data.SkinThickness.isnull().sum()) 
    # 22
    print(data.Insulin.isnull().sum()) 
    #0 
    print(data.BMI.isnull().sum()) 
    # 11 
    print(data.DiabetesPedigreeFunction.isnull().sum())
    # 0 
    print(data.Age.isnull().sum())
    # 19 
    print(data.Outcome.isnull().sum())
    # 0

'''
Before handling missing data checked if this value 
missing becuase it wasn't recorded or becuase it doesn't exist?

For coolumn of 'Preganancies' we assume them to be male and put Null values as 0

Rest of the values need to be updated as they can't be null and thus we assume 
they weren't recorded.
So if the patient has a Null value then and fill the mean 
value of patients with outcome 1 (mean_1)
or mean value of patients with outcome 0 (mean_0) 
according to the outcome of the patient, if patients
if outcome = 1  replace with mean_1
if outcome = 0  replace with mean_0

'''


data['Pregnancies'] = pd.to_numeric(data['Pregnancies'], errors = 'coerce')
data['BloodPressure'] = pd.to_numeric(data['BloodPressure'], errors = 'coerce')
data['SkinThickness'] = pd.to_numeric(data['SkinThickness'], errors = 'coerce')
data['BMI'] = pd.to_numeric(data['BMI'], errors = 'coerce')

data.Pregnancies.fillna(value=0 ,inplace= True)

mean_1 = data[data.Outcome == 1].Glucose.mean()
print(mean_1)

mean_0 = data[data.Outcome == 0].Glucose.mean()
print(mean_0)


data['Glucose'] = data.apply(lambda x: mean_0 if np.isnan(x['Glucose']) and x['Outcome'] == 0 else mean_1 if np.isnan(x['Glucose']) and x['Outcome'] == 1 else x['Glucose'], axis = 1)
data['SkinThickness'] = data.apply(lambda x: mean_0 if np.isnan(x['SkinThickness']) and x['Outcome'] == 0 else mean_1 if np.isnan(x['SkinThickness']) and x['Outcome'] == 1 else x['Glucose'], axis = 1)
data['BMI'] = data.apply(lambda x: mean_0 if np.isnan(x['BMI']) and x['Outcome'] == 0 else mean_1 if np.isnan(x['BMI']) and x['Outcome'] == 1 else x['Glucose'], axis = 1)
data['Age'] = data.apply(lambda x: mean_0 if np.isnan(x['Age']) and x['Outcome'] == 0 else mean_1 if np.isnan(x['Age']) and x['Outcome'] == 1 else x['Glucose'], axis = 1)

data.reset_index()


data.fillna(0, inplace=True)

X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
Y= data['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

logmodel = LogisticRegression()

# Fit the model using the training data
# X_train -> parameter supplies the data features
# y_train -> parameter supplies the target labels
logmodel.fit(X_train, Y_train)

predictions = logmodel.predict(X_test)


print(classification_report(Y_test, predictions))