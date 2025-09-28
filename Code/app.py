import os
import sys
import math
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont
import cv2  # only used to ensure robust image loading when needed
from inference_sdk import InferenceHTTPClient

# === CONFIG ===
API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
API_KEY = os.getenv("ROBOFLOW_API_KEY", "122aOY67jDoRdfvlcYg6")  # <-- put your key or set env ROBOFLOW_API_KEY
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "tb-all-3/1")       # <-- change to your model, e.g., "tb-all-3/1"

# If you run a local inference server, set:
#   set ROBOFLOW_API_URL=http://localhost:9001  (Windows)
#   export ROBOFLOW_API_URL=http://localhost:9001 (Linux/macOS)
# =================

def ask_image_path() -> Path:
    p = input("Enter path to an image (JPG/PNG): ").strip().strip('"')
    path = Path(p)
    if not path.exists():
        print("Error: file not found.")
        sys.exit(1)
    if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        print("Error: please provide a .jpg/.jpeg or .png file.")
        sys.exit(1)
    return path

def load_image_pil(path: Path) -> Image.Image:
    # Use PIL for drawing; cv2 fallback ensures weird encodings still load
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        arr = cv2.imread(str(path))
        if arr is None:
            raise RuntimeError("Unable to load image.")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(arr)

def text_bg(draw: ImageDraw.ImageDraw, xy: Tuple[int,int], text: str,
            fill=(255,255,255), bg=(0,0,0)):
    font = ImageFont.load_default()
    # Get text bounding box (x0,y0,x1,y1)
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = xy
    pad = 2
    draw.rectangle([x, y, x + tw + 2*pad, y + th + 2*pad], fill=bg)
    draw.text((x + pad, y + pad), text, fill=fill, font=font)

def draw_predictions(img: Image.Image, preds: dict) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Roboflow object-detection response usually at preds["predictions"]
    predictions = preds.get("predictions", [])
    for p in predictions:
        # Expected keys: x, y, width, height, confidence, class
        x = p.get("x"); y = p.get("y")
        bw = p.get("width"); bh = p.get("height")
        conf = p.get("confidence", 0.0)
        cls = p.get("class", "object")

        if None in (x, y, bw, bh):
            # Skip if shape info is missing (e.g., segmentation result)
            continue

        # Convert center-based box (x,y,w,h) -> top-left & bottom-right
        left = x - bw / 2
        top = y - bh / 2
        right = x + bw / 2
        bottom = y + bh / 2

        # Clamp to image bounds
        left = max(0, left); top = max(0, top)
        right = min(w - 1, right); bottom = min(h - 1, bottom)

        # Choose a simple color (green); Pillow expects tuples
        color = (0, 255, 0)
        thickness = max(2, math.ceil(min(w, h) * 0.0025))

        # Draw rectangle with thickness
        for t in range(thickness):
            draw.rectangle([left - t, top - t, right + t, bottom + t], outline=color)

        label = f"{cls} {conf:.2f}"
        text_bg(draw, (int(left), int(top) - 14 if top > 14 else int(top) + 2), label)

    return img

def main():
    img_path = ask_image_path()

    print(f"\nUsing API_URL={API_URL}")
    print(f"Model: {MODEL_ID}\n")

    client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

    # Run inference
    try:
        result = client.infer(str(img_path), model_id=MODEL_ID)
    except Exception as e:
        print(f"Inference error: {e}")
        sys.exit(1)

    # Load and draw
    img = load_image_pil(img_path)
    img_out = draw_predictions(img, result)

    out_path = img_path.with_name(f"{img_path.stem}_predicted{img_path.suffix}")
    img_out.save(out_path)
    print(f"Saved: {out_path}")

    # Optional: print raw result for debugging
    # import json; print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
