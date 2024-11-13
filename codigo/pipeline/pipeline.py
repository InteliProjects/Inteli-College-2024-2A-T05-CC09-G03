# Importando as bibliotecas
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd
from collections import deque
from tensorflow.keras.models import load_model
import tensorflow as tf
import easyocr
import re
import os

# Create output directory for saved images
output_images_dir = '/content/detected_heads'
os.makedirs(output_images_dir, exist_ok=True)

# Paths to the color and thermal videos
color_video_path = "/content/drive/MyDrive/approach_nwot/videos/video_colorido_apresentacao.mp4"
thermal_video_path = "/content/drive/MyDrive/approach_nwot/videos/video_termico_apresentacao.mp4"

# Paths to the head detection and eye segmentation models
model_path = '/content/drive/MyDrive/approach_nwot/models/modelo_deteccao_cabeca_completo.pth'
model2_path = '/content/drive/MyDrive/data/saved_model/my_model.keras'

# Define crop intervals for color and thermal frames
color_crop_rows = (0, 720)
color_crop_cols = (250, 500)

# Function to crop the frame
def crop_frame(frame, crop_rows, crop_cols):
    return frame[crop_rows[0]:crop_rows[1], crop_cols[0]:crop_cols[1]]

# Function to load the head detection model
def load_complete_model(model_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model, device

# Load the head detection model
model, device = load_complete_model(model_path)

# Function to define dice loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice

# Load the eye segmentation model
model2_carregado = load_model(model2_path, custom_objects={'dice_loss': dice_loss})

# Initialize video capture
color_cap = cv2.VideoCapture(color_video_path)
thermal_cap = cv2.VideoCapture(thermal_video_path)

# Get original FPS and frame counts
original_fps_color = color_cap.get(cv2.CAP_PROP_FPS)
original_fps_thermal = thermal_cap.get(cv2.CAP_PROP_FPS)

# Desired FPS
desired_fps = 10

# Calculate the frame step
frame_step_color = int(round(original_fps_color / desired_fps))
frame_step_thermal = int(round(original_fps_thermal / desired_fps))

# Get total frame counts
total_frames_color = int(color_cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames_thermal = int(thermal_cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Ensure both videos have the same number of frames after sampling
total_frames = min(total_frames_color // frame_step_color, total_frames_thermal // frame_step_thermal)

# Get frame size from color video
frame_width = int(color_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(color_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
output_video_path = '/content/annotated_output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_video_path, fourcc, desired_fps, (frame_width, frame_height))

# Define block size and calculate the number of blocks
block_size = 50
thermal_height = 512  # After resizing
thermal_width = 640   # After resizing
num_blocks_vert = int(np.ceil(thermal_height / block_size))
num_blocks_hor = int(np.ceil(thermal_width / block_size))

# Create a grid of blocks
blocks = []
for row in range(num_blocks_vert):
    for col in range(num_blocks_hor):
        y_start = row * block_size
        y_end = min((row + 1) * block_size, thermal_height)
        x_start = col * block_size
        x_end = min((col + 1) * block_size, thermal_width)
        blocks.append({
            'row': row,
            'col': col,
            'y_start': y_start,
            'y_end': y_end,
            'x_start': x_start,
            'x_end': x_end,
        })

# Define specific shifts for some blocks
blocks_shifts = {
    (0, 0): {'deltaX': 10, 'deltaY': 35},
    (0, 1): {'deltaX': 20, 'deltaY': 80},
    (0, 2): {'deltaX': 20, 'deltaY': 80},
    (0, 3): {'deltaX': 20, 'deltaY': 80},
    (0, 4): {'deltaX': 20, 'deltaY': 80},
    (1, 0): {'deltaX': 20, 'deltaY': 60},
    (1, 1): {'deltaX': 20, 'deltaY': 80},
    (1, 2): {'deltaX': 20, 'deltaY': 80},
    (1, 3): {'deltaX': 20, 'deltaY': 80},
    (1, 4): {'deltaX': 20, 'deltaY': 80},
    (2, 0): {'deltaX': 20, 'deltaY': 90},
    (2, 1): {'deltaX': 10, 'deltaY': 60},
    (2, 2): {'deltaX': 10, 'deltaY': 60},
    (2, 3): {'deltaX': 20, 'deltaY': 60},
    (2, 4): {'deltaX': 20, 'deltaY': 80},
    (3, 0): {'deltaX': 20, 'deltaY': 80},
    (3, 1): {'deltaX': 10, 'deltaY': 80},
    (3, 2): {'deltaX': 20, 'deltaY': 90},
    (3, 3): {'deltaX': 30, 'deltaY': 90},
    (3, 4): {'deltaX': 30, 'deltaY': 80},
    (4, 0): {'deltaX': 20, 'deltaY': 100},
    (4, 1): {'deltaX': 5, 'deltaY': 110},
    (4, 2): {'deltaX': 15, 'deltaY': 90},
    (4, 3): {'deltaX': 15, 'deltaY': 100},
    (4, 4): {'deltaX': 20, 'deltaY': 100},
    (5, 0): {'deltaX': 20, 'deltaY': 100},
    (5, 1): {'deltaX': 35, 'deltaY': 90},
    (5, 2): {'deltaX': 30, 'deltaY': 100},
    (5, 3): {'deltaX': 25, 'deltaY': 105},
    (5, 4): {'deltaX': 20, 'deltaY': 100},
    (6, 0): {'deltaX': 20, 'deltaY': 100},
    (6, 1): {'deltaX': 20, 'deltaY': 70},
    (6, 2): {'deltaX': 20, 'deltaY': 100},
    (6, 3): {'deltaX': 20, 'deltaY': 80},
    (6, 4): {'deltaX': 40, 'deltaY': 100},
    (7, 0): {'deltaX': 20, 'deltaY': 100},
    (7, 1): {'deltaX': 10, 'deltaY': 85},
    (7, 2): {'deltaX': 10, 'deltaY': 80},
    (7, 3): {'deltaX': 25, 'deltaY': 90},
    (7, 4): {'deltaX': 20, 'deltaY': 100},
    (8, 0): {'deltaX': 20, 'deltaY': 100},
    (8, 1): {'deltaX': 20, 'deltaY': 80},
    (8, 2): {'deltaX': 10, 'deltaY': 100},
    (8, 3): {'deltaX': 10, 'deltaY': 100},
    (8, 4): {'deltaX': 30, 'deltaY': 100},
    (9, 0): {'deltaX': 20, 'deltaY': 100},
    (9, 1): {'deltaX': 20, 'deltaY': 100},
    (9, 2): {'deltaX': 20, 'deltaY': 100},
    (9, 3): {'deltaX': 20, 'deltaY': 100},
    (9, 4): {'deltaX': 30, 'deltaY': 100},
    (10, 0): {'deltaX': 10, 'deltaY': 100},
    (10, 1): {'deltaX': 10, 'deltaY': 100},
    (10, 2): {'deltaX': 10, 'deltaY': 100},
    (10, 3): {'deltaX': 10, 'deltaY': 100},
}

# Function to determine the block of a bounding box
def get_block_of_box(box, blocks):
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2

    for block in blocks:
        if (block['x_start'] <= x_center < block['x_end']) and (block['y_start'] <= y_center < block['y_end']):
            return (block['row'], block['col'])
    return None

# Function to compute IoU between two bounding boxes
def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Function to compute the centroid distance between two bounding boxes
def compute_centroid_distance(box1, box2):
    centroid1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    centroid2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
    return distance

# Initialize variables for tracking
cow_id_counter = 0
cow_ids = {}
tracking_data = []
iou_threshold = 0.3
distance_threshold = 50
conf_threshold = 0.5
history_length = 5
max_age = 3
cow_history = {}
cow_age = {}
max_temperatures = {}

# Function to format and convert OCR text to float
def format_and_convert_to_float(ocr_text):
    cleaned_text = re.sub(r'[^\d]', '', ocr_text)
    if len(cleaned_text) >= 2:
        formatted_text = cleaned_text[:2] + '.' + cleaned_text[2:]
    else:
        formatted_text = cleaned_text + '.0'
    try:
        return float(formatted_text)
    except ValueError:
        return None

# Function to extract temperatures and timestamp from the frame using OCR
def extract_temperatures_from_frame(frame, reader):
    # The frame is expected to be 640x512 after resizing

    # Fixed coordinates for temp_max and temp_min
    roi_temp_max = frame[50:120, 500:575]
    roi_temp_min = frame[400:440, 500:575]

    # Coordinates for timestamp extraction
    roi_timestamp = frame[10:50, 200:350]

    # Check if ROIs are valid
    if roi_temp_max.size == 0 or roi_temp_min.size == 0 or roi_timestamp.size == 0:
        print("ROI for temperature or timestamp extraction is empty.")
        return None, None, None

    # Preprocess ROIs for better OCR accuracy
    roi_temp_max_gray = cv2.cvtColor(roi_temp_max, cv2.COLOR_BGR2GRAY)
    roi_temp_min_gray = cv2.cvtColor(roi_temp_min, cv2.COLOR_BGR2GRAY)

    _, roi_temp_max_thresh = cv2.threshold(roi_temp_max_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, roi_temp_min_thresh = cv2.threshold(roi_temp_min_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    roi_timestamp_gray = cv2.cvtColor(roi_timestamp, cv2.COLOR_BGR2GRAY)
    _, roi_timestamp_thresh = cv2.threshold(roi_timestamp_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use the preprocessed images for OCR
    result_temp_max = reader.readtext(roi_temp_max_thresh)
    result_temp_min = reader.readtext(roi_temp_min_thresh)
    result_timestamp = reader.readtext(roi_timestamp_thresh)

    temp_max_text = ''.join([detection[1] for detection in result_temp_max])
    temp_min_text = ''.join([detection[1] for detection in result_temp_min])
    timestamp_text = ''.join([detection[1] for detection in result_timestamp])

    temp_max = format_and_convert_to_float(temp_max_text)
    temp_min = format_and_convert_to_float(temp_min_text)

    if temp_max is None or temp_min is None:
        temp_max, temp_min = None, None

    timestamp_text = timestamp_text.strip()

    return temp_max, temp_min, timestamp_text

# Initialize the OCR reader
reader = easyocr.Reader(['en'])

# Loop to process the frames of the videos
for i in range(total_frames):
    frame_num_color = i * frame_step_color
    frame_num_thermal = i * frame_step_thermal

    print(f"Processing frame {i} (Color frame {frame_num_color}, Thermal frame {frame_num_thermal})")

    # Set frame positions
    color_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_color)
    thermal_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_thermal)

    ret_color, color_frame = color_cap.read()
    ret_thermal, thermal_frame = thermal_cap.read()

    if not ret_color or not ret_thermal:
        print(f"Failed to read frames at index {i}")
        break

    # Apply crop to the color frame
    color_cropped = crop_frame(color_frame, color_crop_rows, color_crop_cols)

    # Convert the cropped color frame to RGB and tensor
    img_rgb = cv2.cvtColor(color_cropped, cv2.COLOR_BGR2RGB)
    img_tensor = T.ToTensor()(img_rgb).to(device)

    # Head detection on the color frame
    with torch.no_grad():
        detections = model([img_tensor])

    # Extract boxes and scores
    boxes = detections[0]['boxes'].cpu().numpy()
    scores = detections[0]['scores'].cpu().numpy()

    # Filter detections by confidence
    indices = np.where(scores > conf_threshold)[0]
    heads = boxes[indices]

    if len(heads) == 0:
        print(f"No heads detected in frame {i}")
        out_video.write(color_frame)
        continue  # Skip processing this frame

    current_heads = []
    current_ids = []
    matched_ids = set()

    # For each current detection, try to find a match with existing IDs
    for box in heads:
        best_iou = 0
        best_id = None
        best_distance = float('inf')

        for cid, history in cow_history.items():
            for prev_box in list(history)[-history_length:]:
                iou = compute_iou(box, prev_box)
                distance = compute_centroid_distance(box, prev_box)
                if iou > best_iou and distance < best_distance:
                    best_iou = iou
                    best_id = cid
                    best_distance = distance

        if best_iou > iou_threshold and best_distance < distance_threshold and best_id not in matched_ids:
            current_heads.append(box)
            current_ids.append(best_id)
            matched_ids.add(best_id)
            cow_history[best_id].append(box)
            cow_age[best_id] = 0
        else:
            cow_id_counter += 1
            current_heads.append(box)
            current_ids.append(cow_id_counter)
            matched_ids.add(cow_id_counter)
            cow_history[cow_id_counter] = deque(maxlen=history_length)
            cow_history[cow_id_counter].append(box)
            cow_age[cow_id_counter] = 0
            max_temperatures[cow_id_counter] = None  # Initialize as None

    # Increment the age of all IDs not matched in this frame
    for cid in list(cow_age.keys()):
        if cid not in matched_ids:
            cow_age[cid] += 1
            if cow_age[cid] > max_age:
                del cow_history[cid]
                del cow_age[cid]
                del max_temperatures[cid]

    # Update cow_ids with the current IDs
    cow_ids = {cid: box for cid, box in zip(current_ids, current_heads)}

    # Resize the thermal frame to 640x512
    thermal_resized = cv2.resize(thermal_frame, (640, 512))

    # Extract temp_max, temp_min, and timestamp from the resized thermal frame
    temp_max, temp_min, timestamp = extract_temperatures_from_frame(thermal_resized, reader)

    if temp_max is not None and temp_min is not None and timestamp is not None:
        print(f"Frame {i}: temp_max={temp_max}, temp_min={temp_min}, timestamp={timestamp}")
    else:
        print(f"Failed to extract temperatures or timestamp for frame {i}")
        out_video.write(color_frame)
        continue  # Skip processing this frame

    # Prepare to draw annotations on the original color frame (not cropped)
    annotated_frame = color_frame.copy()

    # For each detection, process the eye crop and annotate the frame
    for (x_min, y_min, x_max, y_max), cid in zip(current_heads, current_ids):
        # Map the coordinates back to the original frame size
        x_min_orig = int(x_min + color_crop_cols[0])
        y_min_orig = int(y_min + color_crop_rows[0])
        x_max_orig = int(x_max + color_crop_cols[0])
        y_max_orig = int(y_max + color_crop_rows[0])

        # Draw bounding box on the annotated frame
        cv2.rectangle(annotated_frame, (x_min_orig, y_min_orig), (x_max_orig, y_max_orig), (0, 255, 0), 2)

        # Apply block shifts to align color and thermal frames
        block_key = get_block_of_box([x_min_orig, y_min_orig, x_max_orig, y_max_orig], blocks)
        if block_key is not None and block_key in blocks_shifts:
            deltaX = blocks_shifts[block_key]['deltaX']
            deltaY = blocks_shifts[block_key]['deltaY']
        else:
            deltaX = 0
            deltaY = 0

        # Adjust coordinates for thermal frame
        x_min_th = x_min_orig + deltaX
        y_min_th = y_min_orig + deltaY
        x_max_th = x_max_orig + deltaX
        y_max_th = y_max_orig + deltaY

        # Ensure coordinates are within frame boundaries
        x_min_th = max(0, min(thermal_width - 1, x_min_th))
        x_max_th = max(0, min(thermal_width - 1, x_max_th))
        y_min_th = max(0, min(thermal_height - 1, y_min_th))
        y_max_th = max(0, min(thermal_height - 1, y_max_th))

        # Extract the eye region from the thermal frame
        eye_crop = thermal_resized[int(y_min_th):int(y_max_th), int(x_min_th):int(x_max_th)]
        if eye_crop.size == 0:
            print(f"Empty eye crop for cow ID {cid} in frame {i}. Skipping temperature calculation.")
            continue

        # Resize and preprocess the eye_crop
        eye_crop_resized = cv2.resize(eye_crop, (128, 128)) / 255.0
        eye_crop_resized = np.expand_dims(eye_crop_resized, axis=0)

        # Perform segmentation
        mask = model2_carregado.predict(eye_crop_resized, batch_size=1)
        mask = np.squeeze(mask)
        mask = (mask > 0.5).astype(np.uint8)

        if np.sum(mask) == 0:
            print(f"Empty mask for cow ID {cid} in frame {i}. Skipping temperature calculation.")
            continue

        # Compute the max intensity in the masked area
        gray_eye_crop_resized = cv2.cvtColor((eye_crop_resized[0] * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        masked_eye_crop = np.where(mask == 1, gray_eye_crop_resized, 0)
        max_intensity = np.max(masked_eye_crop)

        if max_intensity <= 0:
            print(f"Invalid max intensity for cow ID {cid} in frame {i}. Skipping temperature calculation.")
            continue

        # Convert intensity to actual temperature
        try:
            temperature = temp_min + (temp_max - temp_min) * (max_intensity / 255.0)

            # Update the maximum temperature recorded for the cow
            if max_temperatures.get(cid) is None or temperature > max_temperatures[cid]:
                max_temperatures[cid] = temperature
        except Exception as e:
            print(f"Error calculating temperature for cow ID {cid} in frame {i}: {e}")
            continue

        # Annotate the frame with cow ID and temperature
        label = f'ID: {cid}, Temp: {temperature:.2f}°C'
        cv2.putText(annotated_frame, label, (x_min_orig, y_min_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Code to Save Cropped Images with Overlays ---

        # Crop the color image to the head bounding box
        cropped_color_head = annotated_frame[y_min_orig:y_max_orig, x_min_orig:x_max_orig]

        # Overlay the bounding box on the thermal image
        thermal_with_box = thermal_resized.copy()
        cv2.rectangle(thermal_with_box, (int(x_min_th), int(y_min_th)), (int(x_max_th), int(y_max_th)), (0, 255, 0), 2)

        # Crop the thermal image to the same region
        cropped_thermal_head = thermal_with_box[int(y_min_th):int(y_max_th), int(x_min_th):int(x_max_th)]

        # Resize both images to the same height
        height = max(cropped_color_head.shape[0], cropped_thermal_head.shape[0])
        if cropped_color_head.shape[0] > 0 and cropped_color_head.shape[1] > 0 and cropped_thermal_head.shape[0] > 0 and cropped_thermal_head.shape[1] > 0:
            resized_color_head = cv2.resize(cropped_color_head, (int(cropped_color_head.shape[1] * height / cropped_color_head.shape[0]), height))
            resized_thermal_head = cv2.resize(cropped_thermal_head, (int(cropped_thermal_head.shape[1] * height / cropped_thermal_head.shape[0]), height))

            # Concatenate images side by side
            concatenated_image = np.hstack((resized_color_head, resized_thermal_head))

            # Save the concatenated image
            image_save_path = os.path.join(output_images_dir, f'frame_{i}_id_{cid}.png')
            cv2.imwrite(image_save_path, concatenated_image)
        else:
            print(f"Invalid cropped images for cow ID {cid} in frame {i}. Skipping image saving.")

    # Annotate the frame with the timestamp
    cv2.putText(annotated_frame, f'Time: {timestamp}', (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Record tracking data
    for box, cid in zip(current_heads, current_ids):
        temperature = max_temperatures.get(cid, None)
        if temperature is None:
            temp_str = 'N/A'
        else:
            temp_str = f'{temperature:.2f}'
        tracking_data.append({
            'frame': i,
            'timestamp': timestamp,
            'id': cid,
            'x1': int(box[0]),
            'y1': int(box[1]),
            'x2': int(box[2]),
            'y2': int(box[3]),
            'temperature': temp_str
        })

    # Write the annotated frame to the output video
    out_video.write(annotated_frame)

# Release video resources
color_cap.release()
thermal_cap.release()
out_video.release()

# Save the tracking data
df = pd.DataFrame(tracking_data)
df.to_excel('/content/output_temperatures_with_timestamp.xlsx', index=False)
print("Temperature data with timestamps has been saved to 'output_temperatures_with_timestamp.xlsx'.")
print(f"Annotated video has been saved to '{output_video_path}'.")
print(f"Head images have been saved to '{output_images_dir}'.")