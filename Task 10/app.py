from flask import Flask, render_template, request, jsonify
import re

app = Flask(__name__)

# --- room data (monthly / nightly rates in PKR) ---
rooms = {
    "shared dorm": {
        "monthly": "Rs. 8,000/month",
        "nightly": "Rs. 300/night",
        "size": "Shared, 6 beds per room",
        "beds": "Single Bunk Bed",
        "type": "Shared Bathroom",
        "guests": 1,
    },
    "standard": {
        "monthly": "Rs. 12,000/month",
        "nightly": "Rs. 400/night",
        "size": "Shared, 2-3 beds per room",
        "beds": "Single Beds",
        "type": "Shared Bathroom",
        "guests": 3,
    },
    "deluxe": {
        "monthly": "Rs. 16,000/month",
        "nightly": "Rs. 550/night",
        "size": "Private Room",
        "beds": "1 Double Bed",
        "type": "Attached Bathroom, AC",
        "guests": 2,
    },
    "private suite": {
        "monthly": "Rs. 22,000/month",
        "nightly": "Rs. 750/night",
        "size": "Private Room + Sitting Area",
        "beds": "1 King Bed",
        "type": "Attached Bathroom, AC, TV",
        "guests": 2,
    },
}

# --- amenity info ---
amenities = {
    "wifi":     "Free Wi-Fi available throughout the hostel. No password needed.",
    "laundry":  "Laundry service available. Rs. 150 per load (wash + dry). Same-day return if dropped before 10 AM.",
    "kitchen":  "Shared kitchen with gas stove, fridge, and basic utensils. Open 24 hours.",
    "canteen":  "On-site canteen serves desi breakfast and dinner. Breakfast Rs. 150 | Dinner Rs. 200. Timings: 7-9 AM and 7-9 PM.",
    "parking":  "Free motorbike parking in the front yard. Street parking available for cars.",
    "security": "24-hour security guard on duty. CCTV cameras cover all common areas.",
    "lounge":   "Common lounge with TV, open 8 AM to midnight for all residents.",
    "study":    "Quiet study room available 9 AM to 11 PM. Proper lighting and table fan included.",
}

# --- policy info ---
policy = {
    "checkin":      "Check-in is from 12:00 PM onward. Early check-in before 12 PM is subject to availability.",
    "checkout":     "Checkout time is 10:00 AM. Late checkout until 12 PM can be arranged with management.",
    "monthly":      "Monthly rent is due on the 1st of each month. 2-day grace period applies. After that, Rs. 200/day late fee.",
    "advance":      "One month advance payment is required at check-in. Fully refundable on departure (minus any dues).",
    "cancellation": "If you cancel before check-in, the advance is refunded minus a Rs. 500 processing fee.",
    "rules":        "Quiet hours: 11 PM to 7 AM. No smoking inside rooms. Valid CNIC required at check-in.",
    "meals":        "Meals are not included in the room rent. The on-site canteen charges separately.",
}

# --- booking info ---
booking = {
    "how":     "To book, call or WhatsApp us at 0300-1234567. You can also visit us directly at Raiwind Rd, Dubai Town Bhobtian, Lahore.",
    "payment": "We accept cash, EasyPaisa, JazzCash, and bank transfer. One month advance is required at check-in.",
    "deposit": "A one-month advance deposit is collected at check-in. Fully refundable when you leave, minus any dues.",
}


def match(text, *words):
    return any(w in text for w in words)


def get_reply(msg):
    txt = msg.lower().strip()
    txt = re.sub(r"[^\w\s]", "", txt)

    # greetings
    if match(txt, "hi", "hello", "hey", "good morning", "good evening", "greetings", "aoa", "assalam"):
        return (
            "Welcome to Dubai Boys Hostel! I can help you with room types, pricing, "
            "facilities, check-in, booking, and hostel rules. What would you like to know?"
        )

    # farewell
    if match(txt, "bye", "goodbye", "see you", "khuda hafiz", "allah hafiz"):
        return "Thank you for reaching out to Dubai Boys Hostel. Hope to see you soon!"

    # thanks
    if match(txt, "thank", "thanks", "shukriya", "appreciate", "helpful"):
        return "You're welcome! Feel free to ask if you have more questions."

    # booking check before room check (avoid "book a room" hitting rooms first)
    if match(txt, "book", "reserve", "reservation", "apply"):
        if match(txt, "payment", "pay", "easypaisa", "jazzcash", "transfer"):
            return booking["payment"]
        if match(txt, "deposit", "advance"):
            return booking["deposit"]
        return booking["how"]

    # rooms
    if match(txt, "room", "rooms", "accommodation", "types", "bed", "space", "suite", "dorm", "deluxe"):
        if match(txt, "dorm", "bunk"):
            r = rooms["shared dorm"]
            return (
                f"Shared Dorm -- {r['monthly']} | {r['nightly']} | "
                f"{r['size']} | {r['beds']} | {r['type']}"
            )
        if match(txt, "standard"):
            r = rooms["standard"]
            return (
                f"Standard Room -- {r['monthly']} | {r['nightly']} | "
                f"{r['size']} | {r['beds']} | {r['type']} | Up to {r['guests']} guests."
            )
        if match(txt, "deluxe"):
            r = rooms["deluxe"]
            return (
                f"Deluxe Room -- {r['monthly']} | {r['nightly']} | "
                f"{r['size']} | {r['beds']} | {r['type']} | Up to {r['guests']} guests."
            )
        if match(txt, "suite", "private"):
            r = rooms["private suite"]
            return (
                f"Private Suite -- {r['monthly']} | {r['nightly']} | "
                f"{r['size']} | {r['beds']} | {r['type']} | Up to {r['guests']} guests."
            )
        lines = ["We have four room options:\n"]
        for name, r in rooms.items():
            lines.append(f"  {name.title()}: {r['monthly']} | {r['nightly']} | {r['type']}")
        lines.append("\nAsk about any specific room for full details.")
        return "\n".join(lines)

    # price
    if match(txt, "price", "cost", "rate", "how much", "kitna", "rent", "monthly", "fee", "charge"):
        if match(txt, "dorm", "bunk"):
            return f"Shared Dorm: {rooms['shared dorm']['monthly']} or {rooms['shared dorm']['nightly']}."
        if match(txt, "standard"):
            return f"Standard Room: {rooms['standard']['monthly']} or {rooms['standard']['nightly']}."
        if match(txt, "deluxe"):
            return f"Deluxe Room: {rooms['deluxe']['monthly']} or {rooms['deluxe']['nightly']}."
        if match(txt, "suite", "private"):
            return f"Private Suite: {rooms['private suite']['monthly']} or {rooms['private suite']['nightly']}."
        return (
            "Monthly rates (per person):\n"
            "  Shared Dorm:   Rs. 8,000\n"
            "  Standard Room: Rs. 12,000\n"
            "  Deluxe Room:   Rs. 16,000\n"
            "  Private Suite: Rs. 22,000\n\n"
            "One month advance required at check-in."
        )

    # wifi
    if match(txt, "wifi", "internet", "network", "connection"):
        return amenities["wifi"]

    # laundry
    if match(txt, "laundry", "wash", "clothes", "washing"):
        return amenities["laundry"]

    # kitchen
    if match(txt, "kitchen", "cook", "cooking", "stove", "fridge"):
        return amenities["kitchen"]

    # food / canteen
    if match(txt, "food", "canteen", "eat", "meal", "dinner", "lunch", "breakfast", "khana"):
        return amenities["canteen"]

    # parking
    if match(txt, "parking", "motorbike", "car", "gaari"):
        return amenities["parking"]

    # security
    if match(txt, "security", "safe", "cctv", "guard"):
        return amenities["security"]

    # lounge / tv
    if match(txt, "lounge", "common", "tv", "television", "sitting"):
        return amenities["lounge"]

    # study room
    if match(txt, "study", "read", "quiet"):
        return amenities["study"]

    # check-in
    if match(txt, "check in", "checkin", "arrive", "arrival"):
        return policy["checkin"]

    # check-out
    if match(txt, "check out", "checkout", "leave", "departure"):
        return policy["checkout"]

    # advance / deposit
    if match(txt, "advance", "deposit"):
        return policy["advance"]

    # cancellation
    if match(txt, "cancel", "refund", "cancellation"):
        return policy["cancellation"]

    # hostel rules
    if match(txt, "rule", "rules", "policy", "smoking", "noise", "quiet hours", "cnic"):
        return policy["rules"]

    # payment
    if match(txt, "payment", "pay", "easypaisa", "jazzcash", "transfer", "cash"):
        return booking["payment"]

    # location / address
    if match(txt, "location", "address", "where", "directions", "kahan", "raiwind"):
        return (
            "Dubai Boys Hostel is located at:\n"
            "Raiwind Rd, Dubai Town Bhobtian, Lahore, Pakistan.\n\n"
            "Call us at 0300-1234567 for exact directions or to arrange pickup."
        )

    # facilities general
    if match(txt, "facilities", "amenities", "services", "offer", "provide"):
        return (
            "Dubai Boys Hostel provides:\n"
            "  Free Wi-Fi | On-site canteen | Shared kitchen\n"
            "  Laundry service | Free motorbike parking\n"
            "  24-hr security & CCTV | Common lounge with TV\n"
            "  Quiet study room\n\n"
            "Ask about any of these for more details."
        )

    # contact / phone
    if match(txt, "contact", "phone", "number", "call", "whatsapp"):
        return "Call or WhatsApp us at 0300-1234567. We respond 9 AM to 10 PM daily."

    # help
    if match(txt, "help", "what can you", "options", "menu"):
        return (
            "I can answer questions about:\n"
            "  Rooms and pricing (monthly and nightly)\n"
            "  Wi-Fi, canteen, kitchen, laundry, parking\n"
            "  Check-in and checkout times\n"
            "  Advance deposit and cancellation policy\n"
            "  Hostel rules and CNIC requirements\n"
            "  How to book and payment methods\n"
            "  Location and contact number\n\n"
            "Type your question and I will answer."
        )

    # fallback
    return (
        "I did not quite get that. Try asking about rooms, pricing, facilities, "
        "check-in, booking, or location. Type 'help' to see all topics."
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"reply": "Please type a message."})
    reply = get_reply(msg)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
