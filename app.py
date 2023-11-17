from flask import Flask, request, jsonify
import joblib
import requests
import os

app = Flask(__name__)

model = joblib.load('model/crop_recommendation_model.joblib')

def kelvinToCelsius(kelvin):
    celsius = kelvin - 273.15
    return celsius

def get_weather_data():
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"
    API_KEY = os.getenv("API_KEY")
    CITY = "Paris"
    url = BASE_URL + "appid=" + API_KEY + "&q=" + CITY
    respone = requests.get(url).json()

    temp_kelvin = respone['main']['temp']
    humidity = respone['main']['humidity']
    temp_celsius = round(kelvinToCelsius(temp_kelvin), 1) 

    return {'temperature': temp_celsius, 'humidity': humidity}

feature_ranges = {
    'N': (0, 100),
    'P': (0, 100),
    'K': (0, 100),
    'temperature': (0, 50),
    'humidity': (0, 100),
    'ph': (0, 14),
    'rainfall': (0, 500)
}

def validate_features(features):
    for feature, (min_val, max_val) in feature_ranges.items():
        if feature not in features:
            return f"Error: {feature} is missing in the input data", 400
        try:
            value = float(features[feature])
        except ValueError:
            return f"Error: {feature} should be a numerical value", 400

        if not (min_val <= value <= max_val):
            return f"Error: {feature} should be in the range [{min_val}, {max_val}]", 400
    return None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    weather_data = get_weather_data()

    validation_result = validate_features({**data, **weather_data})
    if validation_result:
        return jsonify({'error': validation_result[0]}), validation_result[1]

    features = [data['N'], data['P'], data['K'], weather_data['temperature'], weather_data['humidity'], data['ph'], data['rainfall']]

    prediction_probabilities = model.predict_proba([features])[0]

    crop_labels = model.classes_

    crop_suggestions = {}

    for crop_label, probability in zip(crop_labels, prediction_probabilities):
        if probability > 0:
            crop_suggestions[crop_label] = f"{round(probability * 100, 2)}%"
        
    sorted_crop_suggestions = sorted(crop_suggestions.items(), key=lambda x: x[1], reverse=True)

    return jsonify({'crop_suggestions': sorted_crop_suggestions})

if __name__ == '__main__':
    app.run(port=5000)
