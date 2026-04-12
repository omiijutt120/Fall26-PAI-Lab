from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

responses = {
    # Room types
    "room types": "We offer Standard Rooms, Deluxe Rooms, and Presidential Suites.",
    "rooms": "We offer Standard Rooms, Deluxe Rooms, and Presidential Suites.",

    # Prices
    "price": "Standard: PKR 11,000/month | Deluxe: PKR 18,000/month | Suite: PKR 30,000/month.",
    "prices": "Standard: PKR 11,000/month | Deluxe: PKR 18,000/month | Suite: PKR 30,000/month.",
    "cost": "Standard: PKR 11,000/month | Deluxe: PKR 18,000/month | Suite: PKR 30,000/month.",
    "rate": "Standard: PKR 11,000/month | Deluxe: PKR 18,000/month | Suite: PKR 30,000/month.",

    # Amenities
    "amenities": "We provide free WiFi, breakfast, gym, swimming pool, and parking.",
    "wifi": "Yes, free WiFi is available throughout the hotel.",
    "breakfast": "Complimentary breakfast is served from 7:00 AM to 10:30 AM.",
    "pool": "The swimming pool is open from 6:00 AM to 10:00 PM.",
    "gym": "The gym is open 24 hours for all guests.",
    "parking": "Free parking is available for all guests.",

    # Booking
    "book": "You can book online at our website or call +1-800-555-0199.",
    "booking": "You can book online at our website or call +1-800-555-0199.",
    "reserve": "You can book online at our website or call +1-800-555-0199.",
    "reservation": "You can book online at our website or call +1-800-555-0199.",

    # Check in/out
    "check in": "Check-in time is 2:00 PM.",
    "check out": "Check-out time is 12:00 PM (noon).",
    "checkin": "Check-in time is 2:00 PM.",
    "checkout": "Check-out time is 12:00 PM (noon).",

    # Location
    "location": "We are located at Raiwind Rd, Dubai Town, Lahore, Pakistan.",
    "address": "We are located at Raiwind Rd, Dubai Town, Lahore, Pakistan.",
    "where": "We are located at Raiwind Rd, Dubai Town, Lahore, Pakistan.",

    # Contact
    "contact": "Call us at +92-300-5550199 or email info@grandhotel.com.",
    "phone": "Our phone number is +92-300-5550199.",
    "email": "You can email us at info@grandhotel.com.",

    # Greetings
    "hello": "Hello! Welcome to Grand Hotel. How can I help you?",
    "hi": "Hi there! How can I assist you today?",
    "hey": "Hey! How can I help you?",

    # Goodbye
    "bye": "Thank you for visiting Grand Hotel. Have a great day!",
    "goodbye": "Goodbye! We hope to see you soon.",
    "thanks": "You're welcome! Is there anything else I can help you with?",
    "thank you": "You're welcome! Is there anything else I can help you with?",
}

def get_response(message):
    message = message.lower().strip()
    for key in responses:
        if key in message:
            return responses[key]
    return "I'm not sure about that. Please call us at +92-300-5550199 or email info@grandhotel.com."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    reply = get_response(user_message)
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)
