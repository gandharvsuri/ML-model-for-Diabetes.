import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#Load data
data = pd.read_csv("Pima_Indian_diabetes.csv")

print(data.info())

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


def Fill_Null_Values(col):

    null_list = data[data[col].isnull()].index.tolist()
    mean_1 = data[data['Outcome'] == 1][col].mean()
    mean_0 = data[data['Outcome'] == 0][col].mean()




datax= data[data['Outcome'] == 1]
print(data.describe())
#print(data[data['Outcome'] == 0]['Glucose'])

print(data)


Fill_Null_Values('Glucose')
data.Pregnancies.fillna(value=0 ,inplace= True)

mean_1 = data[data.Outcome == 1].Glucose.mean()
print(mean_1)

mean_0 = data[data.Outcome == 0].Glucose.mean()
print(mean_0)

data[data.Outcome == 1].Glucose.fillna(value = mean_1, inplace = True)
data[data.Outcome == 0].Glucose.fillna(value = mean_0, inplace = True)
#data[data.Outcome == 1].Glucose.fillna(value =mean_1 , inplace= True)
#data[data.Outcome == 0].Glucose.fillna(value = mean_0, inplace= True)


data.SkinThickness = data.SkinThickness.reset_index()
SkinThickness_mean = data.SkinThickness.mean()
data.SkinThickness.fillna(value = SkinThickness_mean, inplace = True)

data.BMI= data.BMI.reset_index()
BMI_mean = data.BMI.mean(skipna= True) 
data.BMI.fillna(value = BMI_mean, inplace = True)

Age_mean = data.Age.mean()
data.Age.fillna(value = Age_mean, inplace = True)


print(data)
