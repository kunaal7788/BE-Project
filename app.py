from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Load the saved scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            # Retrieve input data from form
            pregnancies = float(request.form['pregnancies'])
            glucose = float(request.form['glucose'])
            blood_pressure = float(request.form['blood_pressure'])
            skin_thickness = float(request.form['skin_thickness'])
            insulin = float(request.form['insulin'])

            # Prepare the input data
            input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin)
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

            # Standardize the input data using the loaded scaler
            std_data = scaler.transform(input_data_reshaped)

            # Make the prediction
            prediction = classifier.predict(std_data)

            # Determine the result
            result = 'The person is not diabetic' if prediction[0] == 0 else 'The person is diabetic'
        except Exception as e:
            result = f"Error: {e}"

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
