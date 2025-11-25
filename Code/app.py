# app.py — Flask + Roboflow Hosted Inference with Sputum AFB-style report
# ----------------------------------------------------------------------
# How to run:
#   pip install flask inference-sdk pillow opencv-python
#   python app.py
# Then open: http://127.0.0.1:5000/

import math
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from PIL import Image, ImageDraw, ImageFont
import cv2

from inference_sdk import InferenceHTTPClient

# ================== CONFIG ==================
API_URL = "https://serverless.roboflow.com"
API_KEY = "122aOY67jDoRdfvlcYg6"
BASE_MODEL_ID = "tuberculosis-detection-xxmxp/1"

# Thresholds (0.0–1.0)
CONF_THRESHOLD = 0.1
OVERLAP_THRESHOLD = 0.1

# A4 canvas (portrait) at 300 DPI
A4_WIDTH_PX = 2480
A4_HEIGHT_PX = 3508
PAGE_MARGIN = 120  # px

# Folders (under /static so files can be served by Flask easily)
STATIC_DIR = Path("static")
UPLOAD_DIR = STATIC_DIR / "uploads"
OUTPUT_DIR = STATIC_DIR / "outputs"
REPORT_DIR = STATIC_DIR / "reports"
LOGO_PATH = STATIC_DIR / "logo.png"  # save your logo as static/logo.png

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}

for d in [STATIC_DIR, UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
# ============================================

app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret-for-flash-messages"


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix in ALLOWED_EXTENSIONS


def load_image_pil(path: Path):
    """Load an image as PIL.Image in RGB."""
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
    """Draw rectangles for predictions on the image in-place and return it."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    thickness = max(2, math.ceil(min(w, h) * 0.0025))

    for p in preds.get("predictions", []):
        if p.get("confidence", 0.0) < CONF_THRESHOLD:
            continue
        x, y = p.get("x"), p.get("y")
        bw, bh = p.get("width"), p.get("height")
        if None in (x, y, bw, bh):
            continue
        left = max(0, x - bw / 2)
        top = max(0, y - bh / 2)
        right = min(w - 1, x + bw / 2)
        bottom = min(h - 1, y + bh / 2)
        color = (0, 255, 0)
        for t in range(thickness):
            draw.rectangle([left - t, top - t, right + t, bottom + t], outline=color)
    return img


def count_mtb(preds: dict) -> int:
    """Count predictions above confidence threshold."""
    n = 0
    for p in preds.get("predictions", []):
        if p.get("confidence", 0.0) >= CONF_THRESHOLD:
            n += 1
    return n


def classify_afb(mtb_count: int):
    """
    Map MTB count to AFB grade and description, based on the reference scale:
        0  = NO AFB/300 VISUAL FIELDS
        +n = 1-9 AFB/300 VISUAL FIELDS
        1+ = 10-99 AFB/300 VISUAL FIELDS
        2+ = 1-10 AFB/VISUAL FIELD IN AT LEAST 50 FIELDS
        3+ = >10 AFB/VISUAL FIELD IN AT LEAST 20 FIELDS
    A simple thresholding is used for demonstration.
    """
    if mtb_count <= 0:
        grade = "0"
        desc = "NO AFB/300 VISUAL FIELDS"
        diagnosis = "NEGATIVE FOR AFB (0 = NO AFB/300 VISUAL FIELDS)"
    elif 1 <= mtb_count <= 9:
        grade = "+n"
        desc = "1-9 AFB/300 VISUAL FIELDS"
        diagnosis = "AFB POSITIVE, +n (1-9 AFB/300 VISUAL FIELDS)"
    elif 10 <= mtb_count <= 99:
        grade = "1+"
        desc = "10-99 AFB/300 VISUAL FIELDS"
        diagnosis = "AFB POSITIVE, 1+ (10-99 AFB/300 VISUAL FIELDS)"
    elif 100 <= mtb_count <= 199:
        grade = "2+"
        desc = "1-10 AFB/VISUAL FIELD IN AT LEAST 50 FIELDS"
        diagnosis = "AFB POSITIVE, 2+ (1-10 AFB/VISUAL FIELD IN AT LEAST 50 FIELDS)"
    else:
        grade = "3+"
        desc = ">10 AFB/VISUAL FIELD IN AT LEAST 20 FIELDS"
        diagnosis = "AFB POSITIVE, 3+ (>10 AFB/VISUAL FIELD IN AT LEAST 20 FIELDS)"

    return grade, desc, diagnosis


def try_load_font(size: int):
    """Attempt to load a common TrueType font; fall back to default if unavailable."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_wrapped_text(draw: ImageDraw.ImageDraw, text: str, xy, font, fill=(0, 0, 0), max_width=A4_WIDTH_PX - 2 * PAGE_MARGIN, line_spacing=8):
    """Draw multi-line wrapped text within max_width, returns bottom y of the block."""
    if not text:
        return xy[1]
    words = text.split()
    line = ""
    x, y = xy
    for word in words:
        test = f"{line} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        w = bbox[2]
        h = bbox[3]
        if w <= max_width:
            line = test
        else:
            draw.text((x, y), line, font=font, fill=fill)
            y += h + line_spacing
            line = word
    if line:
        bbox = draw.textbbox((0, 0), line, font=font)
        h = bbox[3]
        draw.text((x, y), line, font=font, fill=fill)
        y += h
    return y


def make_report(annotated_img_path: Path, export_type: str, patient: dict, mtb_count: int, remarks: str) -> Path:
    """Compose a single-page A4 report image and save as PNG or PDF."""
    # Base page
    page = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), "white")
    draw = ImageDraw.Draw(page)

    # Fonts
    clinic_font = try_load_font(50)
    clinic_small_font = try_load_font(30)
    title_font = try_load_font(64)
    section_font = try_load_font(40)
    body_font = try_load_font(34)
    small_font = try_load_font(28)
    tiny_font = try_load_font(26)

    # Header and logo
    x = PAGE_MARGIN
    y = PAGE_MARGIN

    logo_width = 0
    logo_height = 0
    if LOGO_PATH.exists():
        try:
            logo = Image.open(LOGO_PATH).convert("RGBA")
            max_logo_height = 220
            lw, lh = logo.size
            scale = min(max_logo_height / lh, 1.0)
            new_size = (max(1, int(lw * scale)), max(1, int(lh * scale)))
            logo = logo.resize(new_size, Image.LANCZOS)
            logo_width, logo_height = logo.size
            page.paste(logo, (x, y), logo)
        except Exception:
            logo_width = 0
            logo_height = 0

    text_x = x + logo_width + (40 if logo_width > 0 else 0)
    text_y = y

    clinic_name = "KIDAPAWAN MEDICAL SPECIALISTS CENTER,INC."
    draw.text((text_x, text_y), clinic_name, font=clinic_font, fill=(0, 0, 0))
    bbox = draw.textbbox((0, 0), clinic_name, font=clinic_font)
    text_y += bbox[3] + 6

    line2 = "Sudapin, Kidapawan City, North Cotabato Tel. No.: (064)-577-1767"
    draw.text((text_x, text_y), line2, font=clinic_small_font, fill=(0, 0, 0))
    bbox = draw.textbbox((0, 0), line2, font=clinic_small_font)
    text_y += bbox[3] + 4

    line3 = "Bacteriology, ARSP Accredited 2024-0112"
    draw.text((text_x, text_y), line3, font=clinic_small_font, fill=(0, 0, 0))
    bbox = draw.textbbox((0, 0), line3, font=clinic_small_font)
    text_y += bbox[3] + 20

    # Main title centered
    sputum_title = "SPUTUM AFB RESULT"
    title_bbox = draw.textbbox((0, 0), sputum_title, font=title_font)
    title_w = title_bbox[2]
    title_h = title_bbox[3]
    title_x = (A4_WIDTH_PX - title_w) // 2
    title_y = max(text_y, y + logo_height + 10)
    draw.text((title_x, title_y), sputum_title, font=title_font, fill=(0, 0, 255))
    y = title_y + title_h + 10

    # Processing timestamp
    timestamp_text = f"Processing Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    draw.text((PAGE_MARGIN, y), timestamp_text, font=small_font, fill=(0, 0, 0))
    bbox = draw.textbbox((0, 0), timestamp_text, font=small_font)
    y += bbox[3] + 30

    # Patient info block
    name = patient.get("name", "").strip()
    age = patient.get("age", "").strip()
    sex = patient.get("sex", "").strip()
    patient_id = patient.get("id", "").strip()

    age_sex_value = ""
    if age or sex:
        age_sex_value = f"{age} / {sex}".strip(" /")

    line = f"NAME: {name}"
    draw.text((PAGE_MARGIN, y), line, font=body_font, fill=(0, 0, 0))

    right_x = A4_WIDTH_PX // 2 + 40
    line_r = f"AGE/SEX: {age_sex_value}"
    draw.text((right_x, y), line_r, font=body_font, fill=(0, 0, 0))
    y += draw.textbbox((0, 0), line, font=body_font)[3] + 10

    line = "PHYSICIAN: ____________________"
    draw.text((PAGE_MARGIN, y), line, font=body_font, fill=(0, 0, 0))

    line_r = "Ward: ____________________"
    draw.text((right_x, y), line_r, font=body_font, fill=(0, 0, 0))
    y += draw.textbbox((0, 0), line, font=body_font)[3] + 10

    line = f"Bacte. No.: {patient_id or '____________________'}"
    draw.text((PAGE_MARGIN, y), line, font=body_font, fill=(0, 0, 0))

    line_r = "DATE: ____________________"
    draw.text((right_x, y), line_r, font=body_font, fill=(0, 0, 0))
    y += draw.textbbox((0, 0), line, font=body_font)[3] + 10

    line = "Lab No.: ____________________"
    draw.text((PAGE_MARGIN, y), line, font=body_font, fill=(0, 0, 0))
    y += draw.textbbox((0, 0), line, font=body_font)[3] + 40

    # AFB classification
    grade, grade_desc, diagnosis = classify_afb(mtb_count)
    reading_text = f"{mtb_count} AFB (model-based count)"

    # Specimen table
    table_x = PAGE_MARGIN + 200
    table_y = y
    table_width = A4_WIDTH_PX - table_x - PAGE_MARGIN
    row_height = 120
    header_height = 100
    num_rows = 3  # Visual Appearance, Reading, Laboratory Diagnosis
    table_height = header_height + row_height * num_rows
    col_left_width = 600  # left column for labels

    # Outer rectangle
    draw.rectangle(
        [table_x, table_y, table_x + table_width, table_y + table_height],
        outline=(0, 0, 0),
        width=3,
    )

    # Horizontal lines
    # Header separator
    draw.line(
        [table_x, table_y + header_height, table_x + table_width, table_y + header_height],
        fill=(0, 0, 0),
        width=3,
    )
    # Row separators
    for i in range(1, num_rows):
        y_line = table_y + header_height + row_height * i
        draw.line(
            [table_x, y_line, table_x + table_width, y_line],
            fill=(0, 0, 0),
            width=2,
        )

    # Vertical separator for label vs specimen column
    draw.line(
        [table_x + col_left_width, table_y, table_x + col_left_width, table_y + table_height],
        fill=(0, 0, 0),
        width=3,
    )

    # Header text
    header_text = "Specimen"
    hbbox = draw.textbbox((0, 0), header_text, font=section_font)
    hw = hbbox[2]
    hh = hbbox[3]
    header_cx = table_x + col_left_width + (table_width - col_left_width) // 2 - hw // 2
    header_cy = table_y + (header_height - hh) // 2
    draw.text((header_cx, header_cy), header_text, font=section_font, fill=(0, 0, 0))

    # Row labels
    row_labels = ["Visual Appearance", "Reading", "Laboratory Diagnosis"]
    row_values = [
        "",  # Visual Appearance, left blank
        reading_text,
        diagnosis,
    ]

    for idx, label in enumerate(row_labels):
        row_top = table_y + header_height + row_height * idx
        label_x = table_x + 20
        label_y = row_top + (row_height - draw.textbbox((0, 0), label, font=body_font)[3]) // 2
        draw.text((label_x, label_y), label, font=body_font, fill=(0, 0, 0))

        value = row_values[idx]
        if value:
            value_x = table_x + col_left_width + 20
            value_y = row_top + 20
            draw_wrapped_text(
                draw,
                value,
                (value_x, value_y),
                font=body_font,
                fill=(0, 0, 0),
                max_width=table_width - col_left_width - 40,
            )

    y = table_y + table_height + 40

    # Remarks block
    draw.text((PAGE_MARGIN, y), "Remarks:", font=body_font, fill=(0, 0, 0))
    y += draw.textbbox((0, 0), "Remarks:", font=body_font)[3] + 10

    remarks_top = y
    remarks_height = 3 * 80
    remarks_box_bottom = remarks_top + remarks_height

    # Three underline lines like the sample
    line_spacing = 80
    for i in range(3):
        ly = remarks_top + i * line_spacing + 50
        draw.line(
            [PAGE_MARGIN, ly, A4_WIDTH_PX - PAGE_MARGIN, ly],
            fill=(0, 0, 0),
            width=2,
        )

    if remarks:
        text_y = remarks_top + 10
        draw_wrapped_text(
            draw,
            remarks,
            (PAGE_MARGIN + 10, text_y),
            font=body_font,
            fill=(0, 0, 0),
            max_width=A4_WIDTH_PX - 2 * PAGE_MARGIN - 20,
        )

    y = remarks_box_bottom + 40

    # Result interpretation box
    ri_box_width = A4_WIDTH_PX - 2 * PAGE_MARGIN
    ri_box_height = 360
    ri_x0 = PAGE_MARGIN
    ri_y0 = y
    ri_x1 = ri_x0 + ri_box_width
    ri_y1 = ri_y0 + ri_box_height

    draw.rectangle([ri_x0, ri_y0, ri_x1, ri_y1], outline=(0, 0, 0), width=3)

    # Title inside interpretation box
    ri_title = "RESULT INTERPRETATION"
    tbbox = draw.textbbox((0, 0), ri_title, font=section_font)
    tw = tbbox[2]
    th = tbbox[3]
    t_x = ri_x0 + (ri_box_width - tw) // 2
    t_y = ri_y0 + 20
    draw.text((t_x, t_y), ri_title, font=section_font, fill=(0, 0, 0))

    ri_lines = [
        "0  = NO AFB/300 VISUAL FIELDS",
        "+n = 1-9 AFB/300 VISUAL FIELDS",
        "1+ = 10-99 AFB/300 VISUAL FIELDS",
        "2+ = 1-10 AFB/VISUAL FIELD IN AT LEAST 50 FIELDS",
        "3+ = >10 AFB/VISUAL FIELD IN AT LEAST 20 FIELDS",
    ]

    text_y_start = t_y + th + 20
    for line in ri_lines:
        draw.text((ri_x0 + 40, text_y_start), line, font=body_font, fill=(0, 0, 0))
        text_y_start += draw.textbbox((0, 0), line, font=body_font)[3] + 6

    y = ri_y1 + 80

    # Footer: performed / verified / pathologist
    footer_y = A4_HEIGHT_PX - PAGE_MARGIN - 80
    label_performed = "PERFORMED BY:"
    label_verified = "VERIFIED BY:"
    label_pathologist = "PATHOLOGIST:"

    draw.text((PAGE_MARGIN, footer_y), label_performed, font=body_font, fill=(0, 0, 0))

    mid_x = A4_WIDTH_PX // 2 - 150
    draw.text((mid_x, footer_y), label_verified, font=body_font, fill=(0, 0, 0))

    right_label_bbox = draw.textbbox((0, 0), label_pathologist, font=body_font)
    right_x = A4_WIDTH_PX - PAGE_MARGIN - right_label_bbox[2]
    draw.text((right_x, footer_y), label_pathologist, font=body_font, fill=(0, 0, 0))

    # Annotated image (optional, put above footer if there is room)
    img_available_height = footer_y - 40 - y
    if img_available_height > 200:
        try:
            annotated = Image.open(annotated_img_path).convert("RGB")
        except Exception:
            annotated = None

        if annotated is not None:
            max_w = A4_WIDTH_PX - 2 * PAGE_MARGIN
            max_h = img_available_height
            aw, ah = annotated.size
            scale = min(max_w / aw, max_h / ah) if aw and ah else 1.0
            new_size = (max(1, int(aw * scale)), max(1, int(ah * scale)))
            annotated_resized = annotated.resize(new_size, Image.LANCZOS)
            img_x = PAGE_MARGIN + (max_w - new_size[0]) // 2
            img_y = y
            page.paste(annotated_resized, (img_x, img_y))

    # Output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"report_{timestamp}"
    if export_type == "png":
        report_path = REPORT_DIR / f"{base}.png"
        page.save(report_path, "PNG")
    else:
        report_path = REPORT_DIR / f"{base}.pdf"
        page.save(report_path, "PDF", resolution=300.0)

    return report_path


@app.route("/", methods=["GET", "POST"])
def index():
    orig_url = None
    out_url = None
    report_url = None

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

        patient = {
            "name": request.form.get("patient_name", "").strip(),
            "id": request.form.get("patient_id", "").strip(),
            "age": request.form.get("patient_age", "").strip(),
            "sex": request.form.get("patient_sex", "").strip(),
        }
        remarks = request.form.get("remarks", "").strip()
        export_type = request.form.get("export_type", "pdf").lower()
        if export_type not in {"pdf", "png"}:
            export_type = "pdf"

        filename = secure_filename(file.filename)
        upload_path = UPLOAD_DIR / filename
        file.save(upload_path)

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
            out_filename = f"{Path(filename).stem}_predicted{Path(filename).suffix.lower()}"
            out_path = OUTPUT_DIR / out_filename
            img_out.save(out_path)
        except Exception as e:
            flash(f"Post-processing failed: {e}")
            return redirect(request.url)

        try:
            mtb_count = count_mtb(result)
            report_path = make_report(out_path, export_type, patient, mtb_count, remarks)
        except Exception as e:
            flash(f"Report generation failed: {e}")
            return redirect(request.url)

        orig_url = url_for("static", filename=f"uploads/{filename}")
        out_url = url_for("static", filename=f"outputs/{out_filename}")
        rel_report = report_path.relative_to(STATIC_DIR).as_posix()
        report_url = url_for("static", filename=rel_report)

    return render_template(
        "index.html",
        orig_url=orig_url,
        out_url=out_url,
        report_url=report_url,
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
