# app.py — Flask + Roboflow Hosted Inference (rectangles only)
# ------------------------------------------------------------
# How to run:
#   pip install flask inference-sdk pillow opencv-python
#   python app.py
# Then open: http://127.0.0.1:5000/

import os
import math
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from PIL import Image, ImageDraw
import cv2

from inference_sdk import InferenceHTTPClient

# ================== CONFIG ==================
API_URL = "https://serverless.roboflow.com"
API_KEY = "122aOY67jDoRdfvlcYg6"          
BASE_MODEL_ID = "tb-all-3-lzjdz/2"       

# Thresholds (0.0–1.0)
CONF_THRESHOLD = 0.1
OVERLAP_THRESHOLD = 0.1

# Folders (under /static so files can be served by Flask easily)
STATIC_DIR = Path("static")
UPLOAD_DIR = STATIC_DIR / "uploads"
OUTPUT_DIR = STATIC_DIR / "outputs"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

for d in [STATIC_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
# ============================================

app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret-for-flash-messages"

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def load_image_pil(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        arr = cv2.imread(str(path))
        if arr is None:
            raise RuntimeError("Unable to load image.")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(arr)

def build_model_id_with_params(model_id: str, conf: float, overlap: float) -> str:
    sep = "&" if "?" in model_id else "?"
    return f"{model_id}{sep}confidence={conf}&overlap={overlap}"

def draw_rectangles_only(img: Image.Image, preds: dict) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for p in preds.get("predictions", []):
        if p.get("confidence", 0.0) < CONF_THRESHOLD:
            continue
        x, y = p.get("x"), p.get("y")
        bw, bh = p.get("width"), p.get("height")
        if None in (x, y, bw, bh):
            continue
        left   = max(0, x - bw / 2)
        top    = max(0, y - bh / 2)
        right  = min(w - 1, x + bw / 2)
        bottom = min(h - 1, y + bh / 2)
        color = (0, 255, 0)
        thickness = max(2, math.ceil(min(w, h) * 0.0025))
        for t in range(thickness):
            draw.rectangle([left - t, top - t, right + t, bottom + t], outline=color)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    orig_url = None
    out_url = None

    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part in the request.")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No image selected.")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload a JPG or PNG.")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        upload_path = UPLOAD_DIR / filename
        file.save(upload_path)

        # --- fix key formatting: trim spaces and encode '+' ---
        safe_key = API_KEY.strip().replace("+", "%2B")

        try:
            client = InferenceHTTPClient(api_url=API_URL, api_key=safe_key)
            model_id = build_model_id_with_params(BASE_MODEL_ID, CONF_THRESHOLD, OVERLAP_THRESHOLD)
            result = client.infer(str(upload_path), model_id=model_id)
        except Exception as e:
            flash(f"Inference failed: {e}")
            return redirect(request.url)

        try:
            img = load_image_pil(upload_path)
            img_out = draw_rectangles_only(img, result)
            out_filename = f"{Path(filename).stem}_predicted{Path(filename).suffix}"
            out_path = OUTPUT_DIR / out_filename
            img_out.save(out_path)
        except Exception as e:
            flash(f"Post-processing failed: {e}")
            return redirect(request.url)

        orig_url = url_for("static", filename=f"uploads/{filename}")
        out_url = url_for("static", filename=f"outputs/{out_filename}")

    return render_template(
        "index.html",
        orig_url=orig_url,
        out_url=out_url,
        conf=CONF_THRESHOLD,
        overlap=OVERLAP_THRESHOLD,
        model=BASE_MODEL_ID
    )

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
