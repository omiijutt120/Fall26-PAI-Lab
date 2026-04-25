# Dubai Boys Hostel — QnA Bot
PAI Lab 12 | MiniLM + FAISS Semantic Search

---

## How to run (follow exactly)

### Step 1 — Open terminal INSIDE this folder
```
cd DubaiBoys
```

### Step 2 — Install dependencies
```
pip install -r requirements.txt
```

### Step 3 — Run the app
```
python app.py
```

First run downloads the MiniLM model (~90MB) and builds the FAISS index automatically.
Both happen only once. After that it loads in seconds.

### Step 4 — Open browser
```
http://127.0.0.1:5000
```

---

## Common mistakes

❌ Running `python app.py` from the WRONG folder → always cd into DubaiBoys first
❌ Using the old hotel_qna app.py → delete that folder, only use this one
❌ Closing the terminal while using the site → keep it open

---

## Folder structure
```
DubaiBoys/
├── app.py
├── requirements.txt
├── README.md
└── templates/
    └── index.html
```

`hotel.index` and `hotel.pkl` are created automatically on first run.
