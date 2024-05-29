# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from mlxtend.plotting import plot_decision_regions
import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import sklearn.model_selection as mod
import sklearn.neighbors as nei
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, precision_recall_curve, average_precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
import operator
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

# loading the diabetes dataset to a pandas DataFrame
df = pd.read_csv('diabetes.csv')

# printing the first 5 rows of the dataset
df.head();

"""## Exploratory Data Analysis (EDA)"""

# number of rows and Columns in this dataset
df.shape

# Get data type for each attribute

df.dtypes

# Information about the dataset
df.info()

# statistical measures of the data
df.describe()

"""#### Features and Target"""

df.columns

"""#### Duplicate Values"""

df.duplicated().sum()

"""#### Missing Values"""

df.isnull().sum()

# Imputation
df_copy = df.copy(deep = True)

df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] =df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

df_copy.isnull().sum()


df['Outcome'].value_counts()

df.groupby('Outcome').mean()

df1=df_copy
df1['Glucose'].fillna(df1['Glucose'].mean(), inplace = True)

df1['BloodPressure'].fillna(df1['BloodPressure'].mean(), inplace = True)

df1['SkinThickness'].fillna(df1['SkinThickness'].median(), inplace = True)

df1['Insulin'].fillna(df1['Insulin'].median(), inplace = True)

df1['BMI'].fillna(df1['BMI'].median(), inplace = True)

df1.isnull().sum()

"""### Plotting Null Count Analysis Plot"""

#### Age

age=sns.FacetGrid(df1,col='Outcome')

"""#### Pregnancies"""

df1.columns

Pregnancies=sns.FacetGrid(df1,col='Outcome')
Pregnancies.map(plt.hist,'Pregnancies')

"""#### DiabetesPedigreeFunction"""

DiabetesPedigreeFunction=sns.FacetGrid(df1,col='Outcome')
DiabetesPedigreeFunction.map(plt.hist,'DiabetesPedigreeFunction')

"""##  Nutritional status based on BMI

Nutritional Status Source: World Health Organization.
"""

Nutritional_status = pd.Series([])

for i in range(len(df1)):
    if df1['BMI'][i] == 0.0:
        Nutritional_status[i]="NA"

    elif df1['BMI'][i] < 18.5:
        Nutritional_status[i]="Underweight"

    elif df1['BMI'][i] < 25:
        Nutritional_status[i]="Normal"

    elif df1['BMI'][i] >= 25 and df1['BMI'][i] < 30:
        Nutritional_status[i]="Overweight"

    elif df1['BMI'][i] >= 30:
        Nutritional_status[i]="Obese"

    else:
        Nutritional_status[i]= df1['BMI'][i]

# Insert new column - Nutritional Status
df1.insert(6, "Nutritional Status", Nutritional_status)
df1.head();
df1['Nutritional Status'].value_counts()

NutritionalStatus=sns.FacetGrid(df1,col='Outcome')
NutritionalStatus.map(plt.hist,'Nutritional Status')
OGTT = pd.Series([])

for i in range(len(df1)):
    if df1['Glucose'][i] == 0.0:
        OGTT [i]="NA"

    elif df1['Glucose'][i] <= 140:
        OGTT [i]="Normal"

    elif df1['Glucose'][i] > 140 and df1['Glucose'][i] <= 198:
        OGTT [i]="Impaired Glucose Tolerance"

    elif df1['Glucose'][i] > 198:
        OGTT [i]="Diabetic Level"

    else:
        OGTT [i]= df1['Glucose'][i]

# Insert new column - Glucose Result
df1.insert(2, "Glucose Result", OGTT)
df1['Glucose Result'].value_counts()
Impaired_Glucose_Tolerance_Diabetic = ((df1 ['Glucose'] > 140 ) & (df1 ['Glucose'] <= 198) & (df1 ['Outcome'] == 1)).sum()
Impaired_Glucose_Tolerance_Diabetic

Normal_Glucose_Diabetic = ((df1 ['Glucose'] != 0 ) & (df1 ['Glucose'] <= 140) & (df1 ['Outcome'] == 1)).sum()
Normal_Glucose_Diabetic

Glucose=sns.FacetGrid(df1,col='Outcome')
Glucose.map(plt.hist,'Glucose')

GlucoseResult=sns.FacetGrid(df1,col='Outcome')
GlucoseResult.map(plt.hist,'Glucose Result')

Percentile_skin_thickness = pd.Series([])
df1['Age'].value_counts()

for i in range(len(df1)):


    if df1["Age"][i] >= 20.0 and df1["Age"][i] <= 79.0:

        if df1["SkinThickness"][i] == 0.0:
            Percentile_skin_thickness[i]=" 0 NA"

        elif df1["SkinThickness"][i] < 11.9:
            Percentile_skin_thickness[i]="1 <P5th"

        elif df1["SkinThickness"][i] == 11.9:
            Percentile_skin_thickness[i]="2 P5th"

        elif df1["SkinThickness"][i] > 11.9 and df1["SkinThickness"][i] < 14.0:
            Percentile_skin_thickness[i]="3 P5th - P10th"

        elif df1["SkinThickness"][i] == 14.0:
            Percentile_skin_thickness[i]="4 P10th"
        elif df1["SkinThickness"][i] > 14.0 and  df1["SkinThickness"][i] < 15.8:
            Percentile_skin_thickness[i]="5 P10th - P15th"

        elif df1["SkinThickness"][i] == 15.8:
            Percentile_skin_thickness[i]="6 P15th"

        elif df1["SkinThickness"][i] > 15.8 and df1["SkinThickness"][i] < 18.0:
            Percentile_skin_thickness[i]="7 P15th - P25th"

        elif df1["SkinThickness"][i] == 18.0:
            Percentile_skin_thickness[i]="8 P25th"

        elif df1["SkinThickness"][i] > 18.0 and df1["SkinThickness"][i] < 23.5:
            Percentile_skin_thickness[i]="9 P25th - P50th"

        elif df1["SkinThickness"][i] == 23.5:
            Percentile_skin_thickness[i]="10 P50th"

        elif df1["SkinThickness"][i] > 23.5 and df1["SkinThickness"][i] < 29.0:
            Percentile_skin_thickness[i]="11 P50th - P75th"

        elif df1["SkinThickness"][i] == 29.0:
            Percentile_skin_thickness[i]="12 P75th"

        elif df1["SkinThickness"][i] > 29.0 and df1["SkinThickness"][i] < 31.9:
            Percentile_skin_thickness[i]="13 P75th - P85th"

        elif df1["SkinThickness"][i] == 31.9:
            Percentile_skin_thickness[i]="14 P85th"

        elif df1["SkinThickness"][i] > 31.9 and df1["SkinThickness"][i] < 33.7:
            Percentile_skin_thickness[i]="15 P85th - P90th"
        elif df1["SkinThickness"][i] == 33.7:
            Percentile_skin_thickness[i]="16 P90th"

        elif df1["SkinThickness"][i] > 33.7 and df1["SkinThickness"][i] < 35.9:
            Percentile_skin_thickness[i]="17 P90th - P95th"

        elif df1["SkinThickness"][i] == 35.9:
            Percentile_skin_thickness[i]="18 P95th"

        elif df1["SkinThickness"][i] > 35.9:
            Percentile_skin_thickness[i]="19 >P95th"
    elif df1["Age"][i] >= 80.0:  #Only 1 woman is 81 years old
        if  df1["SkinThickness"][i] > 31.7:
            Percentile_skin_thickness[i]="20 >P95th"


df1.insert(4, "Percentile skin thickness", Percentile_skin_thickness)

df1.head(5);

# Check number of women x Percentile of skin thickness

df1['Percentile skin thickness'].value_counts()


diabetic_malnourished_st = ((df1 ['SkinThickness'] < 15.8) & (df1 ['Outcome'] == 1)).sum()
print(df1)
diabetic_malnourished_st

diabetic_malnourished_st.mean()

"""* The average of glucose is at the normal range (less than 140 mg/dl)."""

SkinThickness=sns.FacetGrid(df1,col='Outcome')
SkinThickness.map(plt.hist,'SkinThickness')

"""### Blood Pressure"""

df1['BloodPressure'].mean()

df1['BloodPressure'].min()

df1['BloodPressure'].max()

"""* The maximum value of Diastolic Blood Pressure shows that there are a possibility of some women to have hypertension (>90 mmHg)"""

BloodPressure=sns.FacetGrid(df1,col='Outcome')
BloodPressure.map(plt.hist,'BloodPressure')

"""### Insulin"""

df1['Insulin'].mean()

"""* The average value of 2h insulin of the samples show a normal range. (16 to 166 mIU/L)"""

Insulin=sns.FacetGrid(df1,col='Outcome')
Insulin.map(plt.hist,'Insulin')

"""### BMI"""

df1['BMI'].mean()

"""* The average value of BMI indicates obesity (BMI >= 30 kg/m2)"""

BMI=sns.FacetGrid(df1,col='Outcome')

"""## Data Visualization"""

#Skew of attributes distributions
# Select only numeric columns
numeric_df = df1.select_dtypes(include=[np.number])

# Calculate skewness row-wise
skew = numeric_df.skew(axis=1)

encoded_columns = pd.get_dummies(df1[['Glucose Result', 'Nutritional Status']])
df1_encoded = pd.concat([numeric_df, encoded_columns], axis=1)

df1['Percentile skin thickness'] = pd.to_numeric(df1['Percentile skin thickness'], errors='coerce')

# If you want to calculate skew for all numeric columns including newly converted or encoded ones:
df1_final = pd.concat([numeric_df, df1['Percentile skin thickness'], encoded_columns], axis=1)
skew = df1_final.skew(axis=1)

skew

"""* Bell shape curve: Blood Pressure
* Right-Skewed: Age, Insulin, Pregnancies, Diabetes Pedigree Function
* Short IQR: insulin, Diabetes Pedigree Function, Blood Pressure and BMI
* At least 75% of the women:
1. are 25 years old or older
2. have BMI nearly 30 kg/m2
3. have insulin level 100 or more
4. have 1 or more pregnancies
5. have glucose level of 100 mg/dL or more
6. have blood pressure of 60 mmHg or more

### Pairplot
"""


"""# Correlation between all the features

### corr_matrix
"""

# Select only numeric columns
numeric_df = df1.select_dtypes(include=[np.number])

# Calculate the correlation matrix
corr_matrix = numeric_df.corr(method='pearson')

# Use pandas get_dummies to convert categorical variable into dummy/indicator variables
dummy_variables = pd.get_dummies(df1[['Glucose Result', 'Nutritional Status']])
# Combine with the numeric DataFrame
combined_df = pd.concat([numeric_df, dummy_variables], axis=1)

# Attempt to convert percentiles to numeric, coercing errors to NaN
df1['Percentile skin thickness'] = pd.to_numeric(df1['Percentile skin thickness'], errors='coerce')

# Assuming combined_df includes all numeric representations
corr_matrix = combined_df.corr(method='pearson')
print(corr_matrix)

"""* There are no strong correlation between the features. The 'strongest' ones are the following (as expected):
* Age x pregnancies (0.54) - Older women tend to have higher number of pregnancies
* Glucose x insulin (0.41) Glucose x outcome (0.49) - Women that have higher level of glucose tend to have higher level of insulin and have Diabetes
* Skin fold thickness x BMI (0.54) - Women with higher skin fold thickness value have higher BMI (and probably are overweight/obese)
* Negative correlation:
*Diabetes Pedigree Function x Pregnancies (-0.033)
* Blood Pressure x Diabetes Pedigree Function (-0.002)
"""

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot = True)

"""## Separating the Features and Target"""

df1.columns

df1.head();

X = df1.drop(columns=['Outcome', 'Glucose Result', 'Percentile skin thickness', 'Nutritional Status'], axis=1)
Y = df1['Outcome']

X.head();

Y.head();


"""## Data Preprocessing

### Standardization
"""

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)

X_sc = standardized_data

print(X_sc)

"""## Train Test Split"""

X_train, X_test, Y_train, Y_test = train_test_split(X_sc,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X_sc.shape, X_train.shape, X_test.shape)

"""## Training the Model

## K Neighbor Classifier
"""

knn = nei.KNeighborsClassifier(n_neighbors=5)

#training the support vector Machine Classifier
knn.fit(X_train, Y_train)

"""### Accuracy Score"""

# accuracy score on the training data
X_train_prediction = knn.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = knn.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

"""### Evaluation:"""

print(confusion_matrix(Y_test,X_test_prediction ))
print(classification_report(Y_test, X_test_prediction))

"""## Support Vector Machine"""

classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

"""## Model Evaluation

### Accuracy Score
"""

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

"""### Evaluation:"""

print(confusion_matrix(Y_test,X_test_prediction ))
print(classification_report(Y_test, X_test_prediction))

"""## Decision Tree"""

dtree = DecisionTreeClassifier()

dtree.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = dtree.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = dtree.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

"""* model is overfitted.

### Evaluation:
"""

print(confusion_matrix(Y_test,X_test_prediction ))
print(classification_report(Y_test, X_test_prediction))

"""## RandomForest"""

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = rfc.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = rfc.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

"""* model is overfitted.

### Evaluation:
"""

print(confusion_matrix(Y_test,X_test_prediction ))
print(classification_report(Y_test, X_test_prediction))

""" ## XgBoost"""

xgb_model = XGBClassifier(gamma=0)

xgb_model.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = xgb_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = xgb_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

"""* model is overfitted.

### Evaluation:
"""

print(confusion_matrix(Y_test,X_test_prediction ))
print(classification_report(Y_test, X_test_prediction))

"""**SVM is the best model for this prediction since it has an accuracy_score of 0.76.**

## Cross Validation

### Recursive feature elimination
"""

# Define KFold
kf = KFold(n_splits=10, shuffle=False, random_state=None)

skf = StratifiedKFold(n_splits=10, random_state=None)

classifier = svm.SVC(kernel='linear')
classifier = SVC(kernel='linear')  # or any other kernel, but here we need linear

rfecv = RFECV (estimator=classifier,step=1, cv=skf, scoring='accuracy')

rfecv.fit(X,Y)

"""### Feature Importance

**Glucose as a feature is the most important in this datase**
"""

feature_names = X.columns[:10]
feature_names

X1 = X[feature_names]

new_features = list(filter(lambda x: x[1],zip(feature_names, rfecv.support_)))
new_features

"""These are the important features"""

X_new = df1[['Pregnancies','Glucose', 'BloodPressure','SkinThickness','Insulin']]

scaler.fit(X_new)

standardized_data = scaler.transform(X_new)

Xnew_sc = standardized_data

Xnew_sc

X_train, X_test, Y_train, Y_test = train_test_split(Xnew_sc,Y, test_size = 0.2, stratify=Y, random_state=2)

print(Xnew_sc.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear',probability=True)

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

"""### Evaluation:"""

print(confusion_matrix(Y_test,X_test_prediction ))
print(classification_report(Y_test, X_test_prediction))

"""### ROC Curve

ROC AUC is a performance metric that measures the area under the ROC curve. It provides a single scalar value that represents the model's ability to discriminate between positive and negative classes across different thresholds. A value closer to 1 indicates better discrimination, while a value of 0.5 suggests random guessing.
"""

# Obtain predicted probabilities for the positive class (class 1)
out_pred_prob = classifier.predict_proba(X_test)[:, 1]

# Calculate false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(Y_test, out_pred_prob)

ras = roc_auc_score(Y_test, out_pred_prob)
ras

"""**A ROC AUC score of approximately 0.7822 indicates that this SVM classifier has moderate discrimination ability.**

### Precision-recall curve

Average Precision (AP) is a performance metric used to evaluate the quality of a binary classification model, particularly in cases where the classes are imbalanced. It summarizes the precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
"""

precision, recall, thresholds = precision_recall_curve(Y_test, out_pred_prob)
aps = average_precision_score(Y_test, out_pred_prob)
aps

"""**An APS of approximately 0.70 suggests that this SVM classifier achieves a moderate level of precision across all recall levels.**

## Prediction
"""

input_data = (5,100,100,19,1000)
print(scaler.n_features_in_)  # This will tell you the number of features the scaler expects.

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')

"""### Saving Model"""

import pickle

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(classifier, file)

