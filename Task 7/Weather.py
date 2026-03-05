from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# WeatherAPI.com configuration
API_KEY = os.getenv("WEATHER_API_KEY", "6255c469a8b440d3a55183745262602")  # Replace with your actual WeatherAPI key
BASE_URL = "https://api.weatherapi.com/v1/current.json"

def get(city):
    try:
        params = {
            "q": city,
            "key": API_KEY,
            "aqi": "no"  # Set to "yes" if you want air quality data
        }
        r = requests.get(BASE_URL, params=params, timeout=5)
        data = r.json()

        # WeatherAPI returns error in a different format
        if "error" in data:
            return {"error": data["error"].get("message", "City not found")}

        # WeatherAPI has different JSON structure than OpenWeather
        return {
            "city": data["location"]["name"],
            "country": data["location"]["country"],
            "temp": round(data["current"]["temp_c"], 1),  # temp_c = Celsius
            "feels": round(data["current"]["feelslike_c"], 1),  # feelslike_c = Feels like in Celsius
            "humidity": data["current"]["humidity"],
            "pressure": data["current"]["pressure_mb"],  # pressure in millibars (hPa)
            "weather": data["current"]["condition"]["text"],  # Weather condition text
            "desc": data["current"]["condition"]["text"],  # Same as weather for WeatherAPI
            "wind": data["current"]["wind_kph"],  # Wind speed in km/h (WeatherAPI default)
            "icon": data["current"]["condition"]["icon"]  # WeatherAPI provides icon URL
        }
    except Exception as e:
        return {"error": str(e)}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/weather", methods=["POST"])
def weather():
    city = request.json.get("city", "").strip()
    if not city:
        return jsonify({"error": "Please enter a city name"})
    result = get(city)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)