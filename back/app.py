from flask import Flask, request, render_template
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('crop_prediction_model.pkl')

# Define the home route
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    nitrogen = float(request.form['nitrogen'])
    phosphorus = float(request.form['phosphorus'])
    potassium = float(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph_value = float(request.form['ph_value'])
    rainfall = float(request.form['rainfall'])

    # Prepare the input for the model
    input_features = [[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]]

    # Make the prediction
    predicted_crop = model.predict(input_features)

    # Render the result
    return render_template('result.html', crop=predicted_crop[0])

if __name__ == '__main__':
    app.run(debug=True)
