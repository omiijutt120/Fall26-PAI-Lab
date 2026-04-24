# Grand Horizon Hotel — Concierge Chatbot
### PAI Lab 10 | Task 4

A rule-based hotel information chatbot built with **Flask** (backend) and **HTML/CSS/JS** (frontend).

---

## Project Structure

```
hotel_bot/
├── app.py              # Flask server + chatbot logic
├── requirements.txt    # Dependencies
└── templates/
    └── index.html      # Frontend UI
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open in browser
http://127.0.0.1:5000
```

---

## What the Bot Knows

| Topic          | Example Questions                          |
|----------------|--------------------------------------------|
| Rooms          | "What rooms do you have?" / "Tell me about the suite" |
| Pricing        | "How much does a deluxe room cost?"        |
| Pool & Spa     | "What are the pool hours?" / "Can I book a massage?" |
| Dining         | "What restaurants are open?" / "Is breakfast included?" |
| Check-in/out   | "What time is check-in?" / "Can I check out late?" |
| Pets           | "Do you allow dogs?"                       |
| Booking        | "How do I reserve a room?" / "What payment methods do you accept?" |
| Parking        | "Is parking available? How much?"          |
| Wi-Fi          | "Is there free WiFi?"                      |
| Location       | "Where is the hotel?"                      |

---

## Tech Used

- **Flask** — lightweight Python web framework
- **Vanilla JS** — fetch API for async chat requests
- **CSS custom properties** — theming with no external CSS framework
- **Google Fonts** — Cormorant Garamond + Jost

---

*Lab 10 — Programming for Artificial Intelligence*
