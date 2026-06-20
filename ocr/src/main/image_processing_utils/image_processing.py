import cv2
import numpy as np

MAX_OCR_IMAGE_DIM = 1600


def resize_image_if_needed(img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    if max(width, height) <= MAX_OCR_IMAGE_DIM:
        return img
    scale = MAX_OCR_IMAGE_DIM / max(width, height)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def calculate_y_threshold(detections):
    heights = []
    for detection in detections:
        box = detection[0]
        height = abs(box[0][1] - box[2][1])
        heights.append(height)
    avg_height = sum(heights) / len(heights) if heights else 10
    return avg_height * 0.5


def calculate_x_threshold(detections, img_width):
    widths = []
    for detection in detections:
        box = detection[0]
        width = abs(box[1][0] - box[0][0])
        widths.append(width)
    avg_width = sum(widths) / len(widths) if widths else 50
    return max(avg_width * 2, img_width * 0.1)


def detect_tables(detections, y_threshold):
    # Convert detections to a hashable form by making the bounding box a tuple
    hashable_detections = []
    for detection in detections:
        box, text, confidence = detection
        # Convert the bounding box list to a tuple of tuples
        hashable_box = tuple(tuple(coord) for coord in box)
        hashable_detections.append((hashable_box, text, confidence))

    sorted_detections = sorted(hashable_detections, key=lambda x: x[0][0][1])

    tables = []
    current_table = []
    last_y = None

    for detection in sorted_detections:
        box = detection[0]
        y = box[0][1]

        if last_y is None or abs(y - last_y) < y_threshold * 2:
            current_table.append(detection)
        else:
            if len(current_table) > 1:
                tables.append(current_table)
            current_table = [detection]
        last_y = y

    if len(current_table) > 1:
        tables.append(current_table)

    # Use a set to store hashable detections
    table_detections = set()
    for table in tables:
        for detection in table:
            table_detections.add(detection)

    # Filter out table detections
    non_table_detections = [d for d in hashable_detections if d not in table_detections]
    return tables, non_table_detections


def format_table(table, y_threshold):
    rows = []
    current_row = []
    last_y = None

    sorted_table = sorted(table, key=lambda x: (x[0][0][1], x[0][0][0]))

    for detection in sorted_table:
        box, text, confidence = detection
        y = box[0][1]
        text = text.strip()

        if not text:
            continue

        if last_y is None or abs(y - last_y) < y_threshold:
            current_row.append((box[0][0], text))
        else:
            current_row = sorted(current_row, key=lambda x: x[0])
            rows.append([item[1] for item in current_row])
            current_row = [(box[0][0], text)]
        last_y = y

    if current_row:
        current_row = sorted(current_row, key=lambda x: x[0])
        rows.append([item[1] for item in current_row])

    if not rows:
        return []

    max_cols = max(len(row) for row in rows)
    col_widths = [0] * max_cols
    for row in rows:
        for i, word in enumerate(row):
            col_widths[i] = max(col_widths[i], len(word))

    formatted_rows = []
    for row in rows:
        formatted_row = " | ".join(word.ljust(col_widths[i]) for i, word in enumerate(row))
        formatted_rows.append(formatted_row)

    return formatted_rows


def detect_columns(detections, x_threshold):
    sorted_by_x = sorted(detections, key=lambda det: det[0][0][0])

    columns = []
    current_column = []
    last_x = None

    for detection in sorted_by_x:
        box = detection[0]
        x = box[0][0]

        if last_x is None or (x - last_x) < x_threshold:
            current_column.append(detection)
        else:
            columns.append(current_column)
            current_column = [detection]
        last_x = x

    if current_column:
        columns.append(current_column)

    return columns


def post_process_text(text, all_detections, current_detection):
    # Generalize the post-processing to avoid hardcoding
    # 1. Add spaces between concatenated words using a simple heuristic
    processed_text = ""
    i = 0
    while i < len(text):
        processed_text += text[i]
        # Add a space if we transition from lowercase to uppercase
        if i + 1 < len(text) and text[i].islower() and text[i + 1].isupper():
            processed_text += " "
        i += 1

    # 2. Look for nearby currency symbols and associate them with numbers
    if processed_text.replace(".", "").isdigit():  # Check if the text is a number
        current_box = current_detection[0]
        current_x, current_y = current_box[0][0], current_box[0][1]

        for detection in all_detections:
            if detection == current_detection:
                continue
            box, other_text, _ = detection
            other_x, other_y = box[0][0], box[0][1]

            if (abs(other_y - current_y) < 10 and
                    abs(other_x - current_x) < 20 and
                    other_x < current_x and
                    other_text in ["£", "$", "€", "¥"]):
                processed_text = other_text + processed_text
                break
    return processed_text
