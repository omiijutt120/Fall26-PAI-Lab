import os
import pickle
import numpy as np
import faiss
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer

app       = Flask(__name__)
IDX_FILE  = "hotel.index"
DOCS_FILE = "hotel.pkl"

DATA = [
    "Dubai Boys Hostel is at Raiwind Rd, Dubai Town Bhobtian, Lahore, Pakistan.",
    "Call or WhatsApp 0300-1234567 for directions or to book a room.",
    "We have four room types: Shared Dorm, Standard, Deluxe, and Private Suite.",
    "Shared Dorm has 6 bunk beds per room with shared bathroom. Rs. 8,000 per month or Rs. 300 per night.",
    "Standard Room has 2 to 3 single beds with shared bathroom. Rs. 12,000 per month or Rs. 400 per night.",
    "Deluxe Room is a private room with double bed, attached bathroom and AC. Rs. 16,000 per month or Rs. 550 per night.",
    "Private Suite has king bed, sitting area, attached bathroom, AC and TV. Rs. 22,000 per month or Rs. 750 per night.",
    "Monthly rent for standard room is Rs. 12,000 per person.",
    "Monthly rent for shared dorm is Rs. 8,000 per person.",
    "Monthly rent for deluxe room is Rs. 16,000.",
    "Monthly rent for private suite is Rs. 22,000.",
    "One month advance deposit is required at check-in and is fully refundable on departure.",
    "Check-in time is 12:00 PM. Early check-in subject to availability.",
    "Checkout time is 10:00 AM. Late checkout until 12 PM can be arranged.",
    "We accept cash, EasyPaisa, JazzCash, and bank transfer for payment.",
    "Cancellation before check-in gets advance refunded minus Rs. 500 processing fee.",
    "Free Wi-Fi is available throughout the hostel with no password needed.",
    "Laundry service costs Rs. 150 per load including wash and dry.",
    "Shared kitchen has gas stove, fridge and utensils and is open 24 hours.",
    "Canteen serves desi breakfast for Rs. 150 and dinner for Rs. 200.",
    "Canteen timings are 7 to 9 AM for breakfast and 7 to 9 PM for dinner.",
    "Free motorbike parking is available in the front yard.",
    "Street parking is available for cars near the hostel.",
    "24-hour security guard is on duty with CCTV covering all common areas.",
    "Common lounge with TV is open 8 AM to midnight for all residents.",
    "Quiet study room is open 9 AM to 11 PM with lighting and a table fan.",
    "Quiet hours are 11 PM to 7 AM and no smoking is allowed inside rooms.",
    "Valid CNIC is required at check-in for all residents.",
    "Monthly rent is due on the 1st with a two-day grace period. Rs. 200 per day late fee after that.",
    "To book a room visit us at Raiwind Rd Lahore or call 0300-1234567.",
]

print("Loading MiniLM model...")
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


def build():
    print("Building FAISS index...")
    vecs = model.encode(DATA, show_progress_bar=True)
    vecs = np.array(vecs, dtype="float32")
    ix   = faiss.IndexFlatL2(vecs.shape[1])
    ix.add(vecs)
    faiss.write_index(ix, IDX_FILE)
    pickle.dump(DATA, open(DOCS_FILE, "wb"))
    print(f"Done — {ix.ntotal} vectors saved.")
    return ix, DATA


def load():
    if os.path.exists(IDX_FILE) and os.path.exists(DOCS_FILE):
        print("Loading saved index...")
        ix   = faiss.read_index(IDX_FILE)
        data = pickle.load(open(DOCS_FILE, "rb"))
        print(f"Loaded {ix.ntotal} vectors.")
        return ix, data
    return build()


ix, store = load()
print("Flask ready.\n")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True) or {}
    q    = body.get("q", "").strip()
    if not q:
        return jsonify({"error": "empty"}), 400
    vec        = np.array(model.encode([q]), dtype="float32")
    dists, ids = ix.search(vec, 3)
    hits = []
    for i, d in zip(ids[0], dists[0]):
        if i != -1:
            hits.append({"text": store[i], "dist": round(float(d), 2)})
    return jsonify({"q": q, "hits": hits})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
