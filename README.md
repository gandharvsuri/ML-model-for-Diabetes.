# ML-model-for-Diabetes.

A machine learning model to classify whether patients in a dataset have diabetes.

## Problem Statement
Given a dataset of patients' information, predict whether a patient has diabetes.

## Dataset
The dataset is called "Pima_Indian_Diabetes" and is provided in the form of a csv file. It has 9 columns featuring pregnancies, glucose level, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function age and the outcome and records of more than 750 patients for each of the columns.

## Installation Requirements
```
pip3 install numpy
pip3 install pandas
pip3 install scikit-learn 
```

## Data Preprocessing
The provided dataset had a lot of missing values or NULL values. It could be because of one of the two reasons : either the value wasn't recorded or it doesn't exist at all.

### Pregnancies
For the NULL values in this column, we can assume those patients to be male and put 0 for them.




