## Novel Ensembled Approach To Healthcare Data Analysis

This project is a web application that predicts whether a person is diabetic based on input values for blood pressure, skin thickness, insulin, pregnancies, glucose, and BMI. The prediction model is built using scikit-learn, and the web interface is created using Flask.

# Requirements
-Python 3.x
-Flask
-scikit-learn

# Installation

Step 1: Install Requirements
Ensure you have Python installed on your system. You can download it from python.org.

Open a terminal and run the following commands to install Flask and scikit-learn:
- pip install flask scikit-learn

Step 2: Open Terminal
Open a terminal or command prompt on your computer.

Step 3: Run the Application
Navigate to the directory where app.py is located. Run the following command:

python app.py

Step 4: Access the Application
After running the above command, you will see an output in the terminal with a URL.

Step 5: Open the URL in a Browser
Paste the URL into your web browser. You should see the home page of the application.


Step 6: Input Your Values
On the home page, enter the following values:

- Blood Pressure
- Skin Thickness
- Insulin
- Pregnancies
- Glucose
- BMI

Step 7: Click on Predict
After entering the values, click the "Predict" button.

Step 8: View the Result
You will see the prediction result displayed on the page. It will either show "The person is diabetic" or "The person is not diabetic" based on the input values.

# Project Structure
- app.py: The main Python file that contains the Flask application and prediction logic.
- templates/: Directory containing HTML templates for the web interface.
- index.html: The home page template.
- diabetes_prediction : 



# Diabetes-Prediction
Diabetes poses a severe health challenge in India due to its high prevalence and associated complications. By identifying diabetes in its early stages, individuals can adopt lifestyle changes and receive appropriate medical treatment to minimize the risk of complications.

# 🎯Objective:
The primary aim of this project is to accurately identify individuals at risk of diabetes based on different features.

# 🔍 Data Cleaning:
 PIMA Indian Diabetes Dataset(Source: Kaggle).

Dealt with the null values, duplicates, zero values, data type of columns.

# 📊 Exploratory Data Analysis (EDA) :

# 💡 Insights:

- Individuals with diabetes tend to have higher average glucose levels (141.26 mg/dL) compared to those without diabetes with an average glucose level of 109.98 mg/dL. 
-People with diabetes appear to have slightly higher skin thickness and insulin levels compared to those without diabetes, but the difference is not drastic. 
- Individuals with diabetes have a higher average BMI (35.14) compared to those without diabetes (30.30). This suggests a correlation between higher BMI and diabetes risk. 
-The Diabetes Pedigree Function is slightly higher for individuals with diabetes (0.5505) compared to those without diabetes (0.4297). 
- The average age of individuals with diabetes (37.07 years).
- 75% of the women have obese 
- Half of the diabetic women showed normal glucose level.
- The average value of 2h insulin of the samples show a normal range (140(mIU/L))

# ⚙️ Feature Selection: 

Recognizes and chooses important attributes using RFE .These characteristics are essential for accurately predicting diabetes. Glucose as a feature is the most important in this dataset.. 

# 🔄 Data Preprocessing:

Standardized the data to ensure fair comparisons between features.
And, selected relevant features to train the model..

# 🤖 Model Training and Evaluation:

- Implemented various models including KNN, SVM, Random Forest, Decision Tree, and XGBoost.
- Performed Cross Validation using GridSearchCV.
- Found SVM to be the most effective model with an testing accuracy score of 0.76.

# 💡 Model Evaluation:

- Achieved a ROC AUC score of approximately 0.7822, indicating moderate discrimination ability.
- Obtained an Average Precision Score (APS) of approximately 0.70, suggesting a moderate level of precision across all recall levels.

# 🚀 Conclusions: 
Early detection of diabetes plays a critical role in preventing complications, improving health outcomes, reducing healthcare costs, and empowering individuals to take control of their health through timely intervention and management. Hence, it is crucial!🌟

# Notes
- Make sure all required libraries are installed before running the application.
- If you encounter any issues, check the terminal for error messages and ensure all dependencies are correctly installed.


# Authors

- Kunal Digole 72147632F
- Sakshi Birajdar 72147596F
- Hemant Mankar 72147748J
- Rutuj Gangawane 72147644K
