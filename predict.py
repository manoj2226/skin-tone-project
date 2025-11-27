# predict.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os

MODEL_PATH = "models/SkinTone_MobileNetV2_v1.h5"
CLASS_MAP_PATH = "models/class_indices.json"
COLOR_JSON = "colorTones.json"

# load model
model = load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

# load class mapping (index -> label)
with open(CLASS_MAP_PATH, "r") as f:
    idx_to_label = json.load(f)   # e.g. {"0":"dark", "1":"light", ...}

# map raw label to friendly names / keys for JSON
label_mapping = {}

# load makeup data
with open(COLOR_JSON, "r") as f:
    makeup_data = json.load(f)

def preprocess_image(img_path, target_size=(224,224)):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_and_recommend(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    arr = preprocess_image(img_path)
    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    raw_label = idx_to_label[str(idx)]
    mapped = raw_label.lower()   # show only raw tone


    print(f"Image: {img_path}")
    print(f"Predicted class (raw): {raw_label}")
    print(f"Mapped tone: {mapped.title()}")

    recs = makeup_data.get(mapped)
    if not recs:
        print("No recommendations found for", mapped)
        return mapped, None

    print("\nRecommended shades:")
    for cat, items in recs.items():
        print(f"\n{cat.title()}:")
        for x in items:
            print(" -", x.get('name'), x.get('hex'))
    return mapped, recs

if __name__ == "__main__":
    test_image = "customTestImages/3.jpg"  # change to your test image
    if os.path.exists(test_image):
        predict_and_recommend(test_image)
    else:
        print("Put a test image at:", test_image)
