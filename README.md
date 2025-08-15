 Diabetes Prediction using Machine Learning
This project predicts whether a person is likely to have diabetes based on health parameters such as glucose level, blood pressure, BMI, and more. It uses a trained machine learning model deployed via a Flask web application.

🚀 Features
Predicts Positive or Negative diabetes results.

Accepts both integer and decimal values for inputs.

User-friendly web interface built with HTML & Flask.

Lightweight and easy to run locally or on cloud hosting.

🛠️ Technologies Used
Python 3.x

Flask (Web framework)

scikit-learn (Machine learning model)

NumPy (Numerical operations)

HTML/CSS (Frontend)

Joblib (Model serialization)

git clone https://github.com/adityasinghcode/diabetes-prediction-ml.git
cd Diabetes-Prediction-ml

python -m venv myenv

pip install -r requirements.txt

python app.py

http://127.0.0.1:5000

 Example Input & Output
Input:

Pregnancies,	Glucose,	BloodPressure,	SkinThickness,	Insulin,	BMI,	DiabetesPedigreeFunction	,Age,output
2	121	70	30	80	25	 1	32	Negative
5	155	80	35	120	30	1	45   positive

import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# Example features: [Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age]
features = [2, 121, 70, 30, 80, 25, 1, 32]
features_array = np.array(features).reshape(1, -1)

# Predict
result = model.predict(features_array)[0]
print("Prediction:", "Positive" if result == 1 else "Negative")


