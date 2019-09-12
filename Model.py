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


#CHECK FOR MISSING DATA
# 26 print(data.Pregnancies.isnull().sum()) 
#16 print(data.Glucose.isnull().sum())
#0 print(data.BloodPressure.isnull().sum()) 
# 22 print(data.SkinThickness.isnull().sum()) 
#0 print(data.Insulin.isnull().sum()) 
# 11 print(data.BMI.isnull().sum()) 
# 0 print(data.DiabetesPedigreeFunction.isnull().sum())
# 19 print(data.Age.isnull().sum())
# 0 print(data.Outcome.isnull().sum())


data1= data
data.fillna(0,inplace=True)

print(data1.describe(include='all'))

data.Pregnancies= data.Pregnancies.reset_index()
Preganancies_mean = data.Pregnancies.mean()

Glucose_mean=data.Glucose.mean()

data.SkinThickness = data.SkinThickness.reset_index()
SkinThickness_mean = data.SkinThickness.mean()

data.BMI= data.BMI.reset_index()
BMI_mean = data.BMI.mean(skipna= True)

Age_mean = data.Age.mean()

data.Pregnancies.fillna(value = Preganancies_mean, inplace = True)
data.Glucose.fillna(value = Glucose_mean, inplace = True)
data.SkinThickness.fillna(value = SkinThickness_mean, inplace = True)
data.BMI.fillna(value = BMI_mean, inplace = True)
data.Age.fillna(value = Age_mean, inplace = True)


