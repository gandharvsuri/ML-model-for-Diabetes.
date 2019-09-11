import numpy as np
import pandas as pd
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

print(data.BMI.value_counts())