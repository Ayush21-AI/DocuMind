import io
import cv2
import numpy as np
from rapidocr import RapidOCR
from ocr.src.main.image_processing_utils.image_processing import (
    calculate_y_threshold,
    calculate_x_threshold,
    detect_tables,
    detect_columns,
    post_process_text,
    format_table,
)

_ocr_instance = None

def get_ocr_engine() -> RapidOCR:
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = RapidOCR()
    return _ocr_instance

async def extract_text_from_images(stream: io.BytesIO) -> str:
    file_bytes = np.frombuffer(stream.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Unable to decode image from stream.")

    # Initialize OCR engine
    ocr_engine = get_ocr_engine()
    ocr_output = ocr_engine(img)

    # RapidOCR 3.6.0 returns RapidOCROutput object with boxes, txts, scores attributes
    if ocr_output.boxes is None or len(ocr_output.boxes) == 0:
        return "No text detected in the image."

    result = [
        (box.tolist(), txt, score)
        for box, txt, score in zip(ocr_output.boxes, ocr_output.txts, ocr_output.scores)
    ]

    # Calculate thresholds
    y_threshold = calculate_y_threshold(result)
    x_threshold = calculate_x_threshold(result, img.shape[1])

    # Step 1: Detect tables
    tables, non_table_detections = detect_tables(result, y_threshold)

    # Step 2: Detect columns from non-table detections
    columns = detect_columns(non_table_detections, x_threshold)

    # Step 3: Gather all detections (columns + tables)
    all_detections = []
    for column in columns:
        all_detections.extend(column)
    for table in tables:
        all_detections.extend(table)

    # Sort top-down
    all_detections = sorted(all_detections, key=lambda x: x[0][0][1])

    # Step 4: Format into lines
    lines = []
    current_line = []
    last_y = None
    current_table = []
    table_start_y = None

    for detection in all_detections:
        box, text, confidence = detection
        y = box[0][1]
        x = box[0][0]
        text = post_process_text(text.strip(), all_detections, detection)

        if not text:
            continue

        in_table = any(detection in table for table in tables)

        if in_table:
            if table_start_y is None or abs(y - table_start_y) < y_threshold * 2:
                current_table.append(detection)
            else:
                if current_table:
                    lines.extend(format_table(current_table, y_threshold))
                current_table = [detection]
                table_start_y = y
        else:
            if current_table:
                lines.extend(format_table(current_table, y_threshold))
                current_table = []
                table_start_y = None

            if last_y is None or abs(y - last_y) < y_threshold:
                current_line.append((x, text))
            else:
                current_line = sorted(current_line, key=lambda item: item[0])
                lines.append(" ".join(item[1] for item in current_line))
                current_line = [(x, text)]
            last_y = y

    # Final flush
    if current_table:
        lines.extend(format_table(current_table, y_threshold))
    if current_line:
        current_line = sorted(current_line, key=lambda item: item[0])
        lines.append(" ".join(item[1] for item in current_line))

    return "\n".join(lines)
