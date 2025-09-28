# app.py — Roboflow Hosted API (rectangles only)
# -----------------------------------------------
# Usage:
#   1) pip install inference-sdk pillow opencv-python
#   2) python app.py
#   3) Enter: myimage.jpg

import sys
import math
from pathlib import Path

from PIL import Image, ImageDraw
import cv2
from inference_sdk import InferenceHTTPClient

# ====== CONFIG ======
API_URL = "https://serverless.roboflow.com"
API_KEY = "122aOY67jDoRdfvlcYg6"     # <-- your Roboflow API key
BASE_MODEL_ID = "tb-all-3-lzjdz/2"  # <-- your model slug/version

# Thresholds (0.0–1.0)
CONF_THRESHOLD = 0.1     # filter out detections below this confidence
OVERLAP_THRESHOLD = 0.1  # IoU for non-max suppression
# ====================

def ask_image_path() -> Path:
    p = input("Enter path to an image (JPG/PNG): ").strip().strip('"')
    path = Path(p)
    if not path.exists():
        print("Error: file not found."); sys.exit(1)
    if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        print("Error: please provide a .jpg/.jpeg or .png file."); sys.exit(1)
    return path

def load_image_pil(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        arr = cv2.imread(str(path))
        if arr is None:
            raise RuntimeError("Unable to load image.")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(arr)

def build_model_id_with_params(model_id: str, conf: float, overlap: float) -> str:
    # Append ?confidence=...&overlap=... to the model_id
    sep = "&" if "?" in model_id else "?"
    return f"{model_id}{sep}confidence={conf}&overlap={overlap}"

def draw_predictions(img: Image.Image, preds: dict) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for p in preds.get("predictions", []):
        conf = p.get("confidence", 0.0)
        if conf < CONF_THRESHOLD:
            continue

        x, y = p.get("x"), p.get("y")
        bw, bh = p.get("width"), p.get("height")
        if None in (x, y, bw, bh):
            continue

        # Convert center-based box (x,y,w,h) -> box corners
        left   = max(0, x - bw / 2)
        top    = max(0, y - bh / 2)
        right  = min(w - 1, x + bw / 2)
        bottom = min(h - 1, y + bh / 2)

        # Draw rectangle
        color = (0, 255, 0)
        thickness = max(2, math.ceil(min(w, h) * 0.0025))
        for t in range(thickness):
            draw.rectangle([left - t, top - t, right + t, bottom + t], outline=color)

    return img

def main():
    img_path = ask_image_path()

    print(f"\nUsing Hosted API: {API_URL}")
    print(f"Model: {BASE_MODEL_ID}")
    print(f"Confidence Threshold: {CONF_THRESHOLD}")
    print(f"Overlap Threshold: {OVERLAP_THRESHOLD}\n")

    client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
    model_id = build_model_id_with_params(BASE_MODEL_ID, CONF_THRESHOLD, OVERLAP_THRESHOLD)

    try:
        result = client.infer(str(img_path), model_id=model_id)
    except Exception as e:
        print(f"Inference error: {e}")
        sys.exit(1)

    img = load_image_pil(img_path)
    img_out = draw_predictions(img, result)

    out_path = img_path.with_name(f"{img_path.stem}_predicted{img_path.suffix}")
    img_out.save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
