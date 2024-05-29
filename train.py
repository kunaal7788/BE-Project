import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Data preprocessing steps
df_copy = df.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

X = df_copy[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin']]
Y = df_copy['Outcome']

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear', probability=True)
classifier.fit(X_train, Y_train)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(classifier, file)

# Save the scaler to a file
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Accuracy on training and test data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Training data accuracy:', training_data_accuracy)
print('Test data accuracy:', test_data_accuracy)
