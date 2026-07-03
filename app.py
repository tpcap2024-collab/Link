from flask import Flask, request, jsonify, send_file
import requests
import cv2
import numpy as np
import traceback
import time
import threading
import os
import json
from urllib.parse import quote

app = Flask(__name__)


# =========================
# APPSHEET CONFIG
# =========================
APP_ID = "5ebec09a-62dd-4fa9-8f14-830fb104518f"
ACCESS_KEY = "V2-2ZX8p-jmYBx-bH09l-nFTYW-cvV8W-7wNy3-zqOQQ-JvMrp"
TABLE_NAME = "Data TFR"


# =========================
# DEBUG CONFIG
# =========================
DEBUG_DIR = "/tmp"
os.makedirs(DEBUG_DIR, exist_ok=True)


# =========================
# LOCK
# =========================
processed_ids = {}
lock = threading.Lock()

PROCESSED_ID_TTL_SECONDS = 10 * 60
MAX_PROCESSED_IDS = 1000


def cleanup_processed_ids():
    now = time.time()

    expired_ids = [
        row_id
        for row_id, ts in processed_ids.items()
        if now - ts > PROCESSED_ID_TTL_SECONDS
    ]

    for row_id in expired_ids:
        processed_ids.pop(row_id, None)

    if len(processed_ids) > MAX_PROCESSED_IDS:
        sorted_items = sorted(
            processed_ids.items(),
            key=lambda x: x[1]
        )

        overflow = len(processed_ids) - MAX_PROCESSED_IDS

        for row_id, _ in sorted_items[:overflow]:
            processed_ids.pop(row_id, None)


# =========================
# PAYLOAD HELPERS
# =========================
def get_first_value(data, keys):
    for key in keys:
        value = data.get(key)

        if value is not None and str(value).strip() != "":
            return value

    return ""


def normalize_text(value):
    return str(value or "").strip()


def extract_url(value):
    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    if text.startswith("http://") or text.startswith("https://"):
        return text

    try:
        obj = json.loads(text)

        if isinstance(obj, dict):
            url = obj.get("Url") or obj.get("url")

            if url:
                return str(url).strip()

    except Exception:
        pass

    marker = "https://"

    if marker in text:
        start = text.find(marker)

        end_candidates = [
            text.find('"', start),
            text.find("'", start),
            text.find(",", start),
            text.find("}", start)
        ]

        end_candidates = [
            i for i in end_candidates
            if i > start
        ]

        if end_candidates:
            end = min(end_candidates)
            return text[start:end].strip()

        return text[start:].strip()

    return text


# =========================
# DOWNLOAD IMAGE
# =========================
def download_image(url):
    try:
        r = requests.get(
            url,
            timeout=15,
            allow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if r.status_code != 200:
            print("IMAGE HTTP ERROR:", r.status_code)
            return None

        img = cv2.imdecode(
            np.frombuffer(r.content, np.uint8),
            cv2.IMREAD_COLOR
        )

        return img

    except Exception as e:
        print("DOWNLOAD ERROR:", e)
        return None


# =========================
# DEBUG SAVE
# =========================
def save_debug(filename, img):
    try:
        path = os.path.join(DEBUG_DIR, filename)
        ok = cv2.imwrite(path, img)
        print(f"SAVE DEBUG {filename}: {ok}")
        return ok

    except Exception as e:
        print("SAVE DEBUG ERROR:", filename, e)
        return False


# =========================
# CLEAN MASK
# =========================
def clean_mask(mask, min_area_ratio=0.002):
    if mask is None or mask.size == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    result = np.zeros_like(mask)

    min_area = int(mask.size * min_area_ratio)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > min_area:
            result[labels == i] = 255

    return result


# =========================
# INBOUND PALLET GROUP FILTER
# จับเป็นกลุ่มและตัด object ที่ลอย
# =========================
def filter_inbound_pallet_groups(mask):
    if mask is None or mask.size == 0:
        return mask

    h, w = mask.shape[:2]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    filtered = np.zeros_like(mask)

    min_area = h * w * 0.008
    max_top_y = int(h * 0.18)

    candidates = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        if bw <= 0 or bh <= 0:
            continue

        aspect = bw / float(bh)

        # ตัดเส้นยาวบาง เช่น หลังคา / ขอบตู้ / เส้นผนัง
        if bh < h * 0.06 and bw > w * 0.30:
            continue

        # ตัด object ที่อยู่ด้านบนมากและมีขนาดไม่ใหญ่พอ
        if y < max_top_y and area < h * w * 0.035:
            continue

        # รูปทรงที่พอเป็นกลุ่มพาเลท / ตะแกรง / ลัง
        if 0.18 <= aspect <= 9.00:
            candidates.append((i, area, x, y, bw, bh))

    if not candidates:
        return filtered

    candidates = sorted(
        candidates,
        key=lambda item: item[1],
        reverse=True
    )

    main_label, main_area, main_x, main_y, main_w, main_h = candidates[0]

    main_bottom = main_y + main_h
    main_center_x = main_x + main_w / 2

    for i, area, x, y, bw, bh in candidates:
        bottom = y + bh
        center_x = x + bw / 2

        vertical_close = abs(bottom - main_bottom) < h * 0.48
        horizontal_close = abs(center_x - main_center_x) < w * 0.60
        large_enough = area > h * w * 0.020

        if i == main_label or large_enough or (vertical_close and horizontal_close):
            filtered[labels == i] = 255

    return filtered


# =========================
# FILLRATE MODEL
# =========================
def gen_fillrate_outbound(
    img,
    debug=True,
    return_empty=False,
    debug_filename="debug_overlay.jpg",
    roi_mode="outbound"
):

    if img is None or img.size == 0:
        return 0

    orig_h, orig_w = img.shape[:2]
    view_type = "rear" if orig_h > orig_w else "side"

    img = cv2.resize(img, (640, 480))

    if view_type == "side":
        h, w = img.shape[:2]
        target_h = int(w * 9 / 16)
        top = max(0, (h - target_h) // 2)
        img = img[top:top + target_h, :]

    h, w = img.shape[:2]

    print(
        f"VIEW={view_type} "
        f"SIZE={w}x{h} "
        f"ROI_MODE={roi_mode}"
    )

    # =========================
    # ROI
    # =========================
    if roi_mode == "inbound_left":
        y1 = int(h * 0.08)
        y2 = int(h * 0.95)
        x1 = int(w * 0.00)
        x2 = int(w * 1.00)

        roi = img[y1:y2, x1:x2]

    elif roi_mode == "inbound_right":
        y1 = int(h * 0.08)
        y2 = int(h * 0.95)
        x1 = int(w * 0.00)
        x2 = int(w * 1.00)

        roi = img[y1:y2, x1:x2]

    elif view_type == "rear":
        y1 = int(h * 0.18)
        y2 = int(h * 0.82)
        x1 = int(w * 0.15)
        x2 = int(w * 0.85)

        roi = img[y1:y2, x1:x2]

    else:
        y1 = int(h * 0.25)
        y2 = int(h * 0.75)
        x1 = int(w * 0.15)
        x2 = int(w * 0.85)

        roi = img[y1:y2, x1:x2]

    print(
        f"ROI_MODE={roi_mode} "
        f"ROI_X={x1}:{x2} "
        f"ROI_Y={y1}:{y2}"
    )

    if roi.size == 0:
        return 0

    rh, rw = roi.shape[:2]

    container_mask = np.full(
        (rh, rw),
        255,
        dtype=np.uint8
    )

    lab = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    roi_norm = cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )

    hsv = cv2.cvtColor(
        roi_norm,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        roi_norm,
        cv2.COLOR_BGR2GRAY
    )

    gray_blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    h_channel, s_channel, v_channel = cv2.split(hsv)

    v_mean = float(v_channel.mean())
    s_mean = float(s_channel.mean())

    # =========================
    # GRAY WALL MASK
    # =========================
    if roi_mode in ["inbound_left", "inbound_right"]:

        lab_check = cv2.cvtColor(
            roi_norm,
            cv2.COLOR_BGR2LAB
        )

        l_check, a_check, b_check = cv2.split(lab_check)

        a_diff = np.abs(a_check.astype(np.int16) - 128)
        b_diff = np.abs(b_check.astype(np.int16) - 128)

        lab_gray_mask = np.where(
            (l_check > 78) &
            (a_diff < 8) &
            (b_diff < 10),
            255,
            0
        ).astype(np.uint8)

        hsv_gray_mask = cv2.inRange(
            hsv,
            (0, 0, 70),
            (180, 38, 245)
        )

        gray_wall_mask = cv2.bitwise_and(
            lab_gray_mask,
            hsv_gray_mask
        )

        gray_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )

        gray_wall_mask = cv2.morphologyEx(
            gray_wall_mask,
            cv2.MORPH_CLOSE,
            gray_kernel,
            iterations=1
        )

        gray_wall_ratio = cv2.countNonZero(gray_wall_mask) / float(gray_wall_mask.size)

        print(
            f"INBOUND GRAY WALL MASK "
            f"RATIO={gray_wall_ratio:.3f}"
        )

    else:
        gray_wall_mask = np.zeros_like(gray)

    # =========================
    # BACKGROUND MASK
    # =========================
    if roi_mode in ["inbound_left", "inbound_right"]:

        ceiling_mask = np.zeros_like(gray)

        ceiling_cut = int(rh * 0.18)
        ceiling_mask[:ceiling_cut, :] = 255

        edge_for_bg = cv2.Canny(
            gray_blur,
            50,
            140
        )

        edge_density_bg = cv2.blur(
            edge_for_bg.astype(np.float32),
            (21, 21)
        )

        smooth_mask = cv2.inRange(
            edge_density_bg,
            0,
            8
        )

        low_sat_mask = cv2.inRange(
            s_channel,
            0,
            70
        )

        smooth_wall_mask = cv2.bitwise_and(
            smooth_mask,
            low_sat_mask
        )

        inbound_background_mask = cv2.bitwise_or(
            gray_wall_mask,
            smooth_wall_mask
        )

        inbound_background_mask = cv2.bitwise_or(
            inbound_background_mask,
            ceiling_mask
        )

        bg_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 9)
        )

        inbound_background_mask = cv2.morphologyEx(
            inbound_background_mask,
            cv2.MORPH_CLOSE,
            bg_kernel,
            iterations=1
        )

        inbound_background_ratio = cv2.countNonZero(
            inbound_background_mask
        ) / float(inbound_background_mask.size)

        print(
            f"INBOUND BACKGROUND MASK "
            f"RATIO={inbound_background_ratio:.3f}"
        )

    else:
        inbound_background_mask = np.zeros_like(gray)

    # =========================
    # COLOR MASKS
    # =========================
    if roi_mode in ["inbound_left", "inbound_right"]:
        green_mask = cv2.inRange(
            hsv,
            (35, 35, 45),
            (95, 255, 255)
        )
    else:
        green_mask = cv2.inRange(
            hsv,
            (35, 45, 45),
            (95, 255, 255)
        )

    if roi_mode in ["inbound_left", "inbound_right"]:
        brown_mask = cv2.inRange(
            hsv,
            (4, 35, 45),
            (38, 255, 245)
        )
    else:
        brown_mask = cv2.inRange(
            hsv,
            (5, 45, 45),
            (35, 255, 230)
        )

    if roi_mode in ["inbound_left", "inbound_right"]:
        cream_mask = cv2.inRange(
            hsv,
            (10, 10, 80),
            (48, 175, 255)
        )
    else:
        cream_mask = np.zeros_like(brown_mask)

    if roi_mode in ["inbound_left", "inbound_right"]:
        blue_mask = cv2.inRange(
            hsv,
            (82, 18, 45),
            (135, 255, 255)
        )
    else:
        blue_mask = cv2.inRange(
            hsv,
            (85, 35, 35),
            (125, 255, 255)
        )

    red_mask_1 = cv2.inRange(
        hsv,
        (0, 55, 45),
        (12, 255, 255)
    )

    red_mask_2 = cv2.inRange(
        hsv,
        (165, 55, 45),
        (180, 255, 255)
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2
    )

    dark_mask = cv2.inRange(
        hsv,
        (0, 55, 0),
        (180, 255, 65)
    )

    if roi_mode in ["inbound_left", "inbound_right"]:
        dark_mask = np.zeros_like(dark_mask)

    # =========================
    # TEXTURE MASK
    # =========================
    adaptive_texture = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    strong_saturation_mask = cv2.inRange(
        s_channel,
        70,
        255
    )

    strong_low_value_mask = cv2.inRange(
        v_channel,
        0,
        75
    )

    texture_candidate = cv2.bitwise_and(
        strong_saturation_mask,
        strong_low_value_mask
    )

    texture_mask = cv2.bitwise_and(
        adaptive_texture,
        texture_candidate
    )

    edges = cv2.Canny(
        gray_blur,
        40,
        120
    )

    edge_density = cv2.blur(
        edges.astype(np.float32),
        (15, 15)
    )

    edge_mask = cv2.inRange(
        edge_density,
        10,
        255
    )

    texture_mask = cv2.bitwise_and(
        texture_mask,
        edge_mask
    )

    # =========================
    # TOP SUPPRESSION
    # =========================
    top_suppress_mask = np.full(
        (rh, rw),
        255,
        dtype=np.uint8
    )

    if roi_mode in ["inbound_left", "inbound_right"]:
        top_cut_ratio = 0.00
    else:
        top_cut_ratio = 0.12 if view_type == "rear" else 0.16

    top_cut = int(rh * top_cut_ratio)

    top_suppress_mask[:top_cut, :] = 0

    texture_mask = cv2.bitwise_and(
        texture_mask,
        top_suppress_mask
    )

    dark_mask = cv2.bitwise_and(
        dark_mask,
        top_suppress_mask
    )

    # =========================
    # REMOVE BACKGROUND FOR INBOUND
    # =========================
    if roi_mode in ["inbound_left", "inbound_right"]:

        not_background_mask = cv2.bitwise_not(inbound_background_mask)

        green_mask = cv2.bitwise_and(
            green_mask,
            not_background_mask
        )

        brown_mask = cv2.bitwise_and(
            brown_mask,
            not_background_mask
        )

        cream_mask = cv2.bitwise_and(
            cream_mask,
            not_background_mask
        )

        blue_mask = cv2.bitwise_and(
            blue_mask,
            not_background_mask
        )

        red_mask = cv2.bitwise_and(
            red_mask,
            not_background_mask
        )

        dark_mask = cv2.bitwise_and(
            dark_mask,
            not_background_mask
        )

        texture_mask = cv2.bitwise_and(
            texture_mask,
            not_background_mask
        )

    # =========================
    # COMBINE COLOR MASKS
    # =========================
    color_cargo_mask = cv2.bitwise_or(
        green_mask,
        brown_mask
    )

    color_cargo_mask = cv2.bitwise_or(
        color_cargo_mask,
        cream_mask
    )

    color_cargo_mask = cv2.bitwise_or(
        color_cargo_mask,
        blue_mask
    )

    color_cargo_mask = cv2.bitwise_or(
        color_cargo_mask,
        red_mask
    )

    # =========================
    # INBOUND COLOR BOOST
    # =========================
    if roi_mode in ["inbound_left", "inbound_right"]:

        inbound_color_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (7, 7)
        )

        color_cargo_mask = cv2.morphologyEx(
            color_cargo_mask,
            cv2.MORPH_CLOSE,
            inbound_color_kernel,
            iterations=2
        )

        color_cargo_mask = cv2.dilate(
            color_cargo_mask,
            inbound_color_kernel,
            iterations=1
        )

    # =========================
    # INBOUND TEXTURE FILTER
    # =========================
    if roi_mode in ["inbound_left", "inbound_right"]:

        texture_allow_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (21, 21)
        )

        inbound_texture_allow = cv2.dilate(
            color_cargo_mask,
            texture_allow_kernel,
            iterations=1
        )

        texture_mask = cv2.bitwise_and(
            texture_mask,
            inbound_texture_allow
        )

        green_ratio = cv2.countNonZero(green_mask) / float(green_mask.size)
        brown_ratio = cv2.countNonZero(brown_mask) / float(brown_mask.size)
        cream_ratio = cv2.countNonZero(cream_mask) / float(cream_mask.size)
        blue_ratio = cv2.countNonZero(blue_mask) / float(blue_mask.size)
        color_ratio = cv2.countNonZero(color_cargo_mask) / float(color_cargo_mask.size)

        print(
            f"INBOUND COLOR RATIOS "
            f"GREEN={green_ratio:.3f} "
            f"BROWN={brown_ratio:.3f} "
            f"CREAM={cream_ratio:.3f} "
            f"BLUE={blue_ratio:.3f} "
            f"COLOR={color_ratio:.3f}"
        )

    # =========================
    # COMBINE CARGO
    # =========================
    cargo_mask = cv2.bitwise_or(
        color_cargo_mask,
        dark_mask
    )

    cargo_mask = cv2.bitwise_or(
        cargo_mask,
        texture_mask
    )

    cargo_mask = cv2.bitwise_and(
        cargo_mask,
        container_mask
    )

    if roi_mode in ["inbound_left", "inbound_right"]:
        cargo_mask = cv2.bitwise_and(
            cargo_mask,
            cv2.bitwise_not(inbound_background_mask)
        )

    # =========================
    # MORPHOLOGY
    # =========================
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    kernel_medium = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_CLOSE,
        kernel_medium,
        iterations=1
    )

    cargo_mask = cv2.morphologyEx(
        cargo_mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    cargo_mask = clean_mask(
        cargo_mask,
        min_area_ratio=0.005
    )

    if roi_mode in ["inbound_left", "inbound_right"]:
        cargo_mask = filter_inbound_pallet_groups(cargo_mask)

    # =========================
    # CONTOUR FILTER
    # =========================
    contours, _ = cv2.findContours(
        cargo_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_mask = np.zeros_like(cargo_mask)

    min_contour_area = rh * rw * 0.0035

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_contour_area:
            continue

        x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)

        if h_box <= 0 or w_box <= 0:
            continue

        aspect_ratio = w_box / float(h_box)

        if 0.20 <= aspect_ratio <= 6.50:
            cv2.drawContours(
                filtered_mask,
                [cnt],
                -1,
                255,
                thickness=-1
            )

    cargo_mask = filtered_mask

    # =========================
    # FALLBACK
    # =========================
    raw_cargo_ratio = cv2.countNonZero(cargo_mask) / float(container_mask.size)

    if raw_cargo_ratio > 0.95:
        print("WARNING: cargo over-detected, fallback to color only")

        cargo_mask = color_cargo_mask.copy()

        cargo_mask = cv2.bitwise_or(
            cargo_mask,
            dark_mask
        )

        cargo_mask = cv2.bitwise_and(
            cargo_mask,
            container_mask
        )

        if roi_mode in ["inbound_left", "inbound_right"]:
            cargo_mask = cv2.bitwise_and(
                cargo_mask,
                cv2.bitwise_not(inbound_background_mask)
            )

            cargo_mask = filter_inbound_pallet_groups(cargo_mask)

        cargo_mask = cv2.morphologyEx(
            cargo_mask,
            cv2.MORPH_CLOSE,
            kernel_medium,
            iterations=1
        )

        cargo_mask = cv2.morphologyEx(
            cargo_mask,
            cv2.MORPH_OPEN,
            kernel_small,
            iterations=1
        )

        cargo_mask = clean_mask(
            cargo_mask,
            min_area_ratio=0.005
        )

        raw_cargo_ratio = cv2.countNonZero(cargo_mask) / float(container_mask.size)

    # =========================
    # EMPTY MASK
    # =========================
    empty_mask = cv2.bitwise_and(
        container_mask,
        cv2.bitwise_not(cargo_mask)
    )

    # =========================
    # PERSPECTIVE WEIGHT
    # =========================
    y = np.linspace(
        0,
        1,
        rh,
        dtype=np.float32
    ).reshape(rh, 1)

    if view_type == "rear":
        weights = 0.70 + (y ** 1.5) * 1.30
    else:
        weights = 0.80 + (y ** 1.3) * 1.10

    container_score = np.sum(
        (container_mask > 0).astype(np.float32) * weights
    )

    cargo_score = np.sum(
        (cargo_mask > 0).astype(np.float32) * weights
    )

    empty_score = np.sum(
        (empty_mask > 0).astype(np.float32) * weights
    )

    if container_score <= 1e-6:
        return 0

    filled_ratio = cargo_score / container_score
    empty_ratio = empty_score / container_score

    filled_ratio = float(
        np.clip(
            filled_ratio,
            0,
            1
        )
    )

    empty_ratio = float(
        np.clip(
            empty_ratio,
            0,
            1
        )
    )

    # =========================
    # CALIBRATION
    # =========================
    filled_volume = (filled_ratio ** 0.95) * 100
    filled_volume = filled_volume * 0.95

    filled_volume = float(
        np.clip(
            filled_volume,
            0,
            100
        )
    )

    empty_volume = 100 - filled_volume

    if return_empty:
        output_volume = empty_volume
    else:
        output_volume = filled_volume

    output_volume = int(round(output_volume / 5) * 5)
    output_volume = max(0, min(100, output_volume))

    print(
        f"VIEW={view_type} "
        f"ROI_MODE={roi_mode} "
        f"VMEAN={v_mean:.1f} "
        f"SMEAN={s_mean:.1f} "
        f"RAW_CARGO={raw_cargo_ratio:.3f} "
        f"FILLED_RATIO={filled_ratio:.3f} "
        f"EMPTY_RATIO={empty_ratio:.3f} "
        f"RETURN={output_volume}% "
        f"MODE={'EMPTY' if return_empty else 'FILLED'}"
    )

    # =========================
    # DEBUG OUTPUT
    # =========================
    if debug:

        color_layer = roi_norm.copy()

        color_layer[cargo_mask > 0] = (
            0,
            255,
            0
        )

        color_layer[empty_mask > 0] = (
            255,
            0,
            0
        )

        overlay = cv2.addWeighted(
            roi_norm,
            0.85,
            color_layer,
            0.15,
            0
        )

        cargo_contours, _ = cv2.findContours(
            cargo_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay,
            cargo_contours,
            -1,
            (0, 255, 255),
            2
        )

        empty_contours, _ = cv2.findContours(
            empty_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay,
            empty_contours,
            -1,
            (255, 0, 0),
            1
        )

        save_debug(debug_filename, overlay)

    return output_volume


# =========================
# INBOUND FILLRATE MODEL
# =========================
def gen_fillrate_inbound(img, debug=True, return_empty=False, side_name="inbound"):

    print(f"START GEN FILLRATE INBOUND: {side_name}")

    if img is None or img.size == 0:
        print(f"INBOUND IMAGE EMPTY: {side_name}")
        return 0

    try:
        if side_name == "left":
            debug_filename = "debug_left_overlay.jpg"
            roi_mode = "inbound_left"

        elif side_name == "right":
            debug_filename = "debug_right_overlay.jpg"
            roi_mode = "inbound_right"

        else:
            debug_filename = "debug_overlay.jpg"
            roi_mode = "outbound"

        result = gen_fillrate_outbound(
            img,
            debug=debug,
            return_empty=return_empty,
            debug_filename=debug_filename,
            roi_mode=roi_mode
        )

        print(f"END GEN FILLRATE INBOUND: {side_name} RESULT={result}%")

        return result

    except Exception:
        print(f"ERROR GEN FILLRATE INBOUND: {side_name}")
        print(traceback.format_exc())
        return 0


# =========================
# UPDATE APPSHEET
# =========================
def update_appsheet(row_id, volume_text):

    table_name_encoded = quote(TABLE_NAME, safe="")

    url = f"https://api.appsheet.com/api/v2/apps/{APP_ID}/tables/{table_name_encoded}/Action"

    headers = {
        "ApplicationAccessKey": ACCESS_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "Action": "Edit",
        "Rows": [
            {
                "ID": row_id,
                "TFR AI": volume_text,
                "Status": "Done"
            }
        ]
    }

    try:
        print("APPSHEET UPDATE URL:", url)
        print("APPSHEET UPDATE PAYLOAD:", payload)

        r = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=20
        )

        print("APPSHEET STATUS:", r.status_code)
        print("APPSHEET RESPONSE:", r.text[:500])

        return r.status_code, r.text

    except requests.exceptions.Timeout as e:
        print("APPSHEET TIMEOUT:", e)
        return 504, str(e)

    except Exception as e:
        print("APPSHEET ERROR:", e)
        return 500, str(e)


# =========================
# HEALTH CHECK
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "container-fillrate-ai"
    })


# =========================
# DEBUG VIEW
# =========================
@app.route("/debug/<filename>", methods=["GET"])
def debug_file(filename):

    allowed = {
        "debug_overlay.jpg",
        "debug_left_overlay.jpg",
        "debug_right_overlay.jpg"
    }

    if filename not in allowed:
        return jsonify({"error": "file not allowed"}), 403

    path = os.path.join(DEBUG_DIR, filename)

    if not os.path.exists(path):
        return jsonify({"error": "debug file not found"}), 404

    return send_file(path, mimetype="image/jpeg")


@app.route("/debug-list", methods=["GET"])
def debug_list():

    files = [
        "debug_overlay.jpg",
        "debug_left_overlay.jpg",
        "debug_right_overlay.jpg"
    ]

    base_url = request.host_url.rstrip("/")

    return jsonify({
        "status": "ok",
        "files": [
            {
                "file": f,
                "url": f"{base_url}/debug/{f}",
                "exists": os.path.exists(os.path.join(DEBUG_DIR, f))
            }
            for f in files
        ]
    })


# =========================
# API ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    row_id = None

    try:
        data = request.get_json(silent=True)

        print("REQUEST DATA:", data)

        if not data:
            return jsonify({"error": "no json"}), 400

        row_id = normalize_text(
            get_first_value(
                data,
                ["id", "ID", "Id"]
            )
        )

        project = normalize_text(
            get_first_value(
                data,
                ["project", "Project", "PROJECT"]
            )
        )

        project_key = project.lower()

        image_url_raw = get_first_value(
            data,
            [
                "link",
                "Link",
                "LINK",
                "rear_link",
                "Rear Link",
                "link Rear",
                "Link Rear"
            ]
        )

        left_url_raw = get_first_value(
            data,
            [
                "link Left",
                "Link Left",
                "LINK LEFT",
                "link left",
                "left_link",
                "link_left",
                "Left Link",
                "Photo Left Link"
            ]
        )

        right_url_raw = get_first_value(
            data,
            [
                "link Right",
                "Link Right",
                "LINK RIGHT",
                "link right",
                "right_link",
                "link_right",
                "Right Link",
                "Photo Right Link"
            ]
        )

        image_url = extract_url(image_url_raw)
        left_url = extract_url(left_url_raw)
        right_url = extract_url(right_url_raw)

        debug = bool(data.get("debug", True))
        return_empty = bool(data.get("return_empty", False))

        if not row_id:
            return jsonify({"error": "missing id"}), 400

        if not project:
            return jsonify({"error": "missing project"}), 400

        print("ROW ID:", row_id)
        print("PROJECT RAW:", project)
        print("PROJECT KEY:", project_key)
        print("LINK RAW:", image_url_raw)
        print("LINK LEFT RAW:", left_url_raw)
        print("LINK RIGHT RAW:", right_url_raw)
        print("LINK:", image_url)
        print("LINK LEFT:", left_url)
        print("LINK RIGHT:", right_url)

        has_left_right = bool(left_url) and bool(right_url)

        is_inbound = (
            project_key == "inbound"
            or has_left_right
        )

        is_outbound = (
            project_key == "outbound"
            and not is_inbound
        )

        print("ROUTING CHECK:", {
            "is_inbound": is_inbound,
            "is_outbound": is_outbound,
            "has_link": bool(image_url),
            "has_left": bool(left_url),
            "has_right": bool(right_url)
        })

        # =========================
        # DUPLICATE LOCK
        # =========================
        with lock:
            cleanup_processed_ids()

            if row_id in processed_ids:
                return jsonify({
                    "status": "skipped",
                    "id": row_id
                }), 200

            processed_ids[row_id] = time.time()

        # =========================
        # PROJECT = INBOUND
        # =========================
        if is_inbound:

            if not left_url or not right_url:
                with lock:
                    processed_ids.pop(row_id, None)

                return jsonify({
                    "error": "missing inbound images",
                    "required": ["link Left", "link Right"],
                    "project": project,
                    "link Left": left_url,
                    "link Right": right_url
                }), 400

            print("DOWNLOAD INBOUND LEFT:", left_url)

            img_left = download_image(left_url)

            if img_left is None:
                with lock:
                    processed_ids.pop(row_id, None)

                return jsonify({
                    "error": "left image fail",
                    "link Left": left_url
                }), 400

            print("DOWNLOAD INBOUND RIGHT:", right_url)

            img_right = download_image(right_url)

            if img_right is None:
                with lock:
                    processed_ids.pop(row_id, None)

                return jsonify({
                    "error": "right image fail",
                    "link Right": right_url
                }), 400

            left_volume = gen_fillrate_inbound(
                img_left,
                debug=debug,
                return_empty=return_empty,
                side_name="left"
            )

            right_volume = gen_fillrate_inbound(
                img_right,
                debug=debug,
                return_empty=return_empty,
                side_name="right"
            )

            volume = int(round(((left_volume + right_volume) / 2) / 5) * 5)
            volume = max(0, min(100, volume))

            mode = "inbound_fillrate"

            print(
                f"PROJECT=Inbound "
                f"LEFT={left_volume}% "
                f"RIGHT={right_volume}% "
                f"AVG={volume}%"
            )

        # =========================
        # PROJECT = OUTBOUND
        # =========================
        elif is_outbound:

            if not image_url:
                with lock:
                    processed_ids.pop(row_id, None)

                return jsonify({
                    "error": "missing outbound image",
                    "required": ["link"],
                    "link": image_url
                }), 400

            print("DOWNLOAD OUTBOUND:", image_url)

            img = download_image(image_url)

            if img is None:
                with lock:
                    processed_ids.pop(row_id, None)

                return jsonify({
                    "error": "image fail",
                    "link": image_url
                }), 400

            volume = gen_fillrate_outbound(
                img,
                debug=debug,
                return_empty=return_empty,
                debug_filename="debug_overlay.jpg",
                roi_mode="outbound"
            )

            mode = "outbound_fillrate"

            print(
                f"PROJECT=Outbound "
                f"VOLUME={volume}% "
                f"MODE={mode}"
            )

        else:
            with lock:
                processed_ids.pop(row_id, None)

            return jsonify({
                "error": "invalid project",
                "project": project,
                "allowed": ["Inbound", "Outbound"]
            }), 400

        volume_text = f"{volume}%"

        print("FINAL VOLUME:", volume_text)

        app_status, app_response = update_appsheet(row_id, volume_text)

        if app_status < 200 or app_status >= 300:
            with lock:
                processed_ids.pop(row_id, None)

            return jsonify({
                "error": "appsheet update fail",
                "appsheet_status": app_status,
                "appsheet_response": app_response[:500],
                "id": row_id,
                "project": project,
                "volume": volume_text
            }), 500

        base_url = request.host_url.rstrip("/")

        return jsonify({
            "status": "success",
            "id": row_id,
            "project": project,
            "volume": volume_text,
            "mode": mode,
            "debug": debug,
            "inbound": {
                "left_link": left_url,
                "right_link": right_url
            } if is_inbound else None,
            "debug_urls": {
                "overlay": f"{base_url}/debug/debug_overlay.jpg",
                "left_overlay": f"{base_url}/debug/debug_left_overlay.jpg",
                "right_overlay": f"{base_url}/debug/debug_right_overlay.jpg",
                "list": f"{base_url}/debug-list"
            }
        })

    except Exception:
        print(traceback.format_exc())

        if row_id:
            with lock:
                processed_ids.pop(row_id, None)

        return jsonify({"error": "server error"}), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000,
        threaded=True
    )
