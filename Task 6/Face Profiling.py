from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import traceback

app = Flask(__name__)

# Load classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

print(f"Face classifier loaded: {not face_cascade.empty()}")
print(f"Eye classifier loaded: {not eye_cascade.empty()}")

def to_python_int(value):
    """Convert numpy int to Python int for JSON serialization"""
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value

def detect_face_multiple_attempts(gray_img):
    """Try multiple detection strategies"""
    attempts = [
        {'scale': 1.1, 'neighbors': 3, 'min': (100, 100)},
        {'scale': 1.05, 'neighbors': 2, 'min': (80, 80)},
        {'scale': 1.2, 'neighbors': 5, 'min': (50, 50)},
        {'scale': 1.01, 'neighbors': 1, 'min': (60, 60)},
    ]

    for attempt in attempts:
        faces = face_cascade.detectMultiScale(
            gray_img,
            scaleFactor=attempt['scale'],
            minNeighbors=attempt['neighbors'],
            minSize=attempt['min']
        )
        if len(faces) > 0:
            return faces, attempt

    # Try with rotated image (for tilted faces)
    for angle in [15, -15, 30, -30]:
        center = (gray_img.shape[1]//2, gray_img.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray_img, M, (gray_img.shape[1], gray_img.shape[0]))
        faces = face_cascade.detectMultiScale(rotated, 1.05, 2, minSize=(80, 80))
        if len(faces) > 0:
            return faces, {'rotated': angle}

    return [], None

def analyze_image(img_data):
    """Analyze image and return results"""
    try:
        # Decode base64
        if ',' in img_data:
            img_data = img_data.split(',')[1]

        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))

        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Try detection
        faces, method = detect_face_multiple_attempts(gray)

        if len(faces) == 0:
            return {
                "success": False,
                "error": "No face detected. Tips: Use good lighting, face the camera directly, remove glasses, or try a different angle."
            }

        # Get largest face - CONVERT TO PYTHON INT
        largest = max(faces, key=lambda f: int(f[2]) * int(f[3]))
        x, y, fw, fh = int(largest[0]), int(largest[1]), int(largest[2]), int(largest[3])

        # Get face ROI
        face_roi = gray[y:y+fh, x:x+fw]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 5, minSize=(20, 20))

        # Calculate measurements - ALL CONVERTED TO PYTHON TYPES
        measurements = {
            "face_width": int(fw),
            "face_height": int(fh),
            "ratio": float(round(fw/fh, 2)),
            "eyes_found": int(len(eyes))
        }

        # Generate personality profile
        traits = []

        # Trait 1: Openness
        if measurements["ratio"] > 1.1:
            traits.append({"trait": "Openness", "level": "High", "emoji": "🎨", "desc": "Creative & curious", "color": "#FF6B9D"})
        else:
            traits.append({"trait": "Openness", "level": "Balanced", "emoji": "🎯", "desc": "Practical & reliable", "color": "#4ECDC4"})

        # Trait 2: Extraversion
        if len(eyes) >= 2:
            traits.append({"trait": "Extraversion", "level": "High", "emoji": "🔥", "desc": "Outgoing & social", "color": "#FFE66D"})
        else:
            traits.append({"trait": "Extraversion", "level": "Calm", "emoji": "🌙", "desc": "Thoughtful & deep", "color": "#95E1D3"})

        # Trait 3: Confidence
        traits.append({"trait": "Confidence", "level": "High", "emoji": "👑", "desc": "Self-assured leader", "color": "#F38181"})

        # Trait 4: Energy
        brightness = float(np.mean(gray[y:y+fh, x:x+fw]))
        if brightness > 100:
            traits.append({"trait": "Energy", "level": "High", "emoji": "⚡", "desc": "Vibrant & enthusiastic", "color": "#FCBAD3"})
        else:
            traits.append({"trait": "Energy", "level": "Calm", "emoji": "🧘", "desc": "Peaceful & balanced", "color": "#A8D8EA"})

        # Generate code
        code = ""
        code += "E" if traits[1]["level"] == "High" else "I"
        code += "N" if traits[0]["level"] == "High" else "S"
        code += "F" if traits[2]["level"] == "High" else "T"
        code += "J" if traits[3]["level"] == "High" else "P"

        # Draw on image
        result_img = img_cv.copy()
        cv2.rectangle(result_img, (x, y), (x+fw, y+fh), (255, 105, 180), 3)
        cv2.putText(result_img, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 105, 180), 2)

        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(result_img, (x+int(ex), y+int(ey)), (x+int(ex)+int(ew), y+int(ey)+int(eh)), (0, 255, 127), 2)

        # Convert to base64
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        buffered = BytesIO()
        result_pil.save(buffered, format="PNG")
        result_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "success": True,
            "measurements": measurements,
            "traits": traits,
            "personality_code": code,
            "method_used": str(method),
            "processed_image": f"data:image/png;base64,{result_base64}"
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {"success": False, "error": f"Processing error: {str(e)}"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "No image provided"})

        result = analyze_image(data["image"])
        return jsonify(result)

    except Exception as e:
        print(f"Route error: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
