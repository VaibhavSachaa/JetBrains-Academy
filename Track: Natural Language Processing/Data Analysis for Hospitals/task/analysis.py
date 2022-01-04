# write your code here
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 8)
general = pd.read_csv("test/general.csv")
prenatal = pd.read_csv("test/prenatal.csv")
sports = pd.read_csv("test/sports.csv")

# print(general.head(20))
# print(prenatal.head(20))
# print(sports.head(20))

prenatal.columns = general.columns.values
sports.columns = general.columns.values

df = pd.concat([general, prenatal, sports], ignore_index=True)
df.drop(columns='Unnamed: 0', inplace=True)
# Task 2
# print(df.sample(20, random_state=30))

# ------ Task 3 ------

# 6. Delete all the empty rows
df.dropna(how='all', inplace=True)
# df.reset_index(drop=True, inplace=True)

# 7 Correct all the gender column values to f and m respectively
# df['gender'].unique()
df['gender'] = df['gender'].map({'man': 'm', 'male': 'm', 'female': 'f'})

# 8 Replace the empty gender column values for prenatal patients with f (we can assume that the prenatal treats only women).
df['gender'].fillna('f', inplace=True)

# 9 Replace the NaN values in the bmi, diagnosis, blood_test, ecg, ultrasound, mri, xray, children, months columns with zeros
features = ['bmi', 'diagnosis', 'blood_test', 'ecg', 'ultrasound', 'mri', 'xray', 'children', 'months']

for feature in features:
    df[feature].fillna(0, inplace=True)

# Print shape of the resulting data frame
# print(df.shape)

# 11 Print random 20 rows of the resulting data frame. For the reproducible output set random_state=30
# print(df.sample(20, random_state=30))

# Stage 4/5: The statistics
# Q1. Which hospital has the highest number of patients?
hospital_with_max_patients = df.hospital.value_counts(ascending = False).index[0]

# Q2. What share of the patients in the general hospital suffers from stomach-related issues? Round the result to the third decimal place.
general_patients_with_stomach_issue = round(df[df['hospital'] == 'general']['diagnosis'].value_counts(normalize=True).loc['stomach'], 3)

# Q3. What share of the patients in the sports hospital suffers from dislocation-related issues? Round the result to the third decimal place.
sports_patients_with_dislocation_issue = round(df[df['hospital'] == 'sports']['diagnosis'].value_counts(normalize=True).loc['dislocation'], 3)

# Q4. What is the difference in the median ages of the patients in the general and sports hospitals?
gs_median_age_diff = df.groupby('hospital')['age'].median().loc['general'] - df.groupby('hospital')['age'].median().loc['sports']

# Q5. In which hospital the blood test was taken the most often (there is the biggest number of t in the blood_test column among all the hospitals)? How many blood tests were taken?
# blood_test column has three values: t= a blood test was taken, f= a blood test wasn't taken, and 0= there is no information.
max_blood_test = df[df['blood_test'] == 't'].groupby('hospital')['blood_test'].count().max()
hospital_with_max_blood_test = df[df['blood_test'] == 't'].groupby('hospital')['blood_test'].count().sort_values(ascending=False).index[0]

# print("The answer to the 1st question is", hospital_with_max_patients)
# print("The answer to the 2nd question is", general_patients_with_stomach_issue)
# print("The answer to the 3rd question is", sports_patients_with_dislocation_issue)
# print("The answer to the 4th question is", gs_median_age_diff)
# print("The answer to the 5th question is", hospital_with_max_blood_test + ",", max_blood_test, "blood tests")

# Stage 5/5: Visualize it_
# Q1. What is the most common age of a patient among all hospitals? Plot a histogram and choose one of the following age ranges: 0-15, 15-35, 35-55, 55-70, or 70-80
df.plot(y='age', kind='hist', bins=[0, 15, 35, 55, 70, 80])
plt.show()

print("The answer to the 1st question: 15-35")

# Q2. What is the most common diagnosis among patients in all hospitals? Create a pie chart
df['diagnosis'].value_counts().plot(kind='pie')
plt.show()

print("The answer to the 2nd question: pregnancy")

# Q3.
plt.violinplot(df['height'])
plt.show()

print("The answer to the 3rd question: It's because of the different age groups")
