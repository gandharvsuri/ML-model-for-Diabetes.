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


Glucose_mean_1 = data[data.Outcome == 1].Glucose.mean()
Glucose_mean_0 = data[data.Outcome == 0].Glucose.mean()

BMI_mean_1 = data[data.Outcome == 1].BMI.mean()
BMI_mean_0 = data[data.Outcome == 0].BMI.mean()

SkinThickness_mean_1 = data[data.Outcome == 1].SkinThickness.mean()
SkinThickness_mean_0 = data[data.Outcome == 0].SkinThickness.mean()

Age_mean_1 = data[data.Outcome == 1].Age.mean()
Age_mean_0 = data[data.Outcome == 0].Age.mean()

BloodPressure_mean_1 = data[data.Outcome == 1].BloodPressure.mean()
BloodPressure_mean_0 = data[data.Outcome == 0].BloodPressure.mean()


data['Glucose'] = data.apply(lambda x: Glucose_mean_0 if np.isnan(x['Glucose']) and x['Outcome'] == 0 else Glucose_mean_1 if np.isnan(x['Glucose']) and x['Outcome'] == 1 else x['Glucose'], axis = 1)
data['SkinThickness'] = data.apply(lambda x: SkinThickness_mean_0 if np.isnan(x['SkinThickness']) and x['Outcome'] == 0 else SkinThickness_mean_1 if np.isnan(x['SkinThickness']) and x['Outcome'] == 1 else x['SkinThickness'], axis = 1)
data['BMI'] = data.apply(lambda x: BMI_mean_0 if np.isnan(x['BMI']) and x['Outcome'] == 0 else BMI_mean_1 if np.isnan(x['BMI']) and x['Outcome'] == 1 else x['BMI'], axis = 1)
data['Age'] = data.apply(lambda x: Age_mean_0 if np.isnan(x['Age']) and x['Outcome'] == 0 else Age_mean_1 if np.isnan(x['Age']) and x['Outcome'] == 1 else x['Age'], axis = 1)
data['BloodPressure'] = data.apply(lambda x: BloodPressure_mean_0 if np.isnan(x['BloodPressure']) and x['Outcome'] == 0 else BloodPressure_mean_1 if np.isnan(x['BloodPressure']) and x['Outcome'] == 1 else x['Age'], axis = 1)


data.reset_index()

print(Glucose_mean_0)
print(BMI_mean_0)

print(data)

X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
Y= data['Outcome']

'''
Diabetic = data.loc[Y == 1]
Non_Diabetic = data.loc[Y == 0]

plt.scatter(Diabetic.iloc[:, 0], Diabetic.iloc[:, 1], s=10, label='Diabetic')
plt.scatter(Non_Diabetic.iloc[:, 0], Non_Diabetic.iloc[:, 1], s=10, label='Non_Diabetic')
plt.legend()
plt.show()
'''

Accuracy = []

for i in range(1000):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    # logmodel = LogisticRegression()
    logmodel = LogisticRegression(solver='liblinear')
    # Fit the model using the training data
    # X_train -> parameter supplies the data features
    # y_train -> parameter supplies the target labels
    logmodel.fit(X_train, Y_train)
    predictions = logmodel.predict(X_test)

    score = logmodel.score(X_test, Y_test)
    Accuracy.append(score)

print(max(Accuracy))
print(min(Accuracy))
print(sum(Accuracy)/len(Accuracy))    

