# app.py

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics
import warnings
import pickle
# Ensure 'convert.py' and 'feature.py' are in the same directory
# or properly installed as modules.
from convert import convertion
from feature import FeatureExtraction

warnings.filterwarnings('ignore')

gbc = None # Initialize model variable to None

# --- Load the model ---
# IMPORTANT: Make sure 'newmodel.pkl' is in the same directory as this app.py file.
try:
    with open("newmodel.pkl", "rb") as file:
        gbc = pickle.load(file)
    print("Model 'newmodel.pkl' loaded successfully.")
except FileNotFoundError:
    print("Error: 'newmodel.pkl' not found. Please ensure your model file is in the correct directory.")
    # You might want to log this error more formally or handle it differently
except pickle.UnpicklingError as e:
    print(f"Error: Could not unpickle 'newmodel.pkl'. The file might be corrupted or saved with an incompatible Python/library version: {e}")
    print("Please ensure your scikit-learn version matches the one used to save the model.")
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/result', methods=['POST', 'GET'])
def predict():
    # Check if the model was loaded successfully
    if gbc is None:
        return render_template("index.html", name="Error: Prediction model not loaded. Check server logs.")

    if request.method == "POST":
        url = request.form["name"]
        try:
            # Instantiate FeatureExtraction object with the URL
            obj = FeatureExtraction(url)
            # Get features list and reshape for model prediction
            x = np.array(obj.getFeaturesList()).reshape(1, -1)

            # Make prediction using the loaded model
            y_pred = gbc.predict(x)[0]
            
            # Use the convertion function
            name = convertion(url, int(y_pred))

            return render_template("index.html", name=name)
        except Exception as e:
            # Catch any errors during feature extraction or prediction
            print(f"Error during prediction for URL '{url}': {e}")
            return render_template("index.html", name=f"Prediction Error: {e}. Please check the URL format or server logs.")
    # If it's a GET request to /result, just redirect to home or show an error
    return render_template("index.html")

@app.route('/usecases', methods=['GET', 'POST'])
def usecases():
    return render_template('usecases.html')

if __name__ == "__main__":
    app.run(debug=True)
