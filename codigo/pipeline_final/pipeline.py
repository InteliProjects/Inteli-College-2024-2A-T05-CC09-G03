#Código feito para Google Colab: https://colab.research.google.com/drive/1ySkRQOtfyGd61Wkj8jw8fHuAuyi3Ndvs#scrollTo=FhqbMDEsnz0e
from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics
!pip install opencv-python
!pip install easyocr
!pip install tqdm

import cv2
import numpy as np
import os
from ultralytics import YOLO
from IPython.display import display, Video
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pandas as pd
import re
from tqdm import tqdm
import easyocr
from google.colab.patches import cv2_imshow


video_path = '/content/drive/MyDrive/video_definitivamente_pequeno.mp4'

model = YOLO('/content/drive/MyDrive/data/detect_cow/weights/best.pt')

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

model2_carregado = load_model('/content/drive/MyDrive/data/saved_model/my_model.keras',
                              custom_objects={'dice_loss': dice_loss})

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

duration_in_seconds = total_frames / fps

delta_seconds = 1.0

timestamps = np.arange(0, duration_in_seconds, delta_seconds)

frames = []

frame_output_dir = '/content/frames'
os.makedirs(frame_output_dir, exist_ok=True)

for idx, t in enumerate(timestamps):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    frame_filename = os.path.join(frame_output_dir, f'frame_{idx}.png')
    cv2.imwrite(frame_filename, frame)

cap.release()

output_dir = '/content/cropped_images'
os.makedirs(output_dir, exist_ok=True)

frames_de_boi = []

for i, frame in enumerate(frames):
    results = model.predict(frame)
    if results:
        frames_de_boi.append(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cropped_image = frame[y1:y2, x1:x2]
            crop_filename = os.path.join(output_dir, f'cropped_{i}_{x1}_{y1}.png')
            cv2.imwrite(crop_filename, cropped_image)

height, width = 128, 128

image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]

subframes_list = []
subframes_mask = []

for image_file in image_files:
    image_path = os.path.join(output_dir, image_file)
    image = load_img(image_path, target_size=(height, width))
    image_array = img_to_array(image)
    subframes_list.append(image_array)
    image_array_scaled = image_array / 255.0
    image_array_scaled = np.expand_dims(image_array_scaled, axis=0)
    mask = model2_carregado.predict(image_array_scaled, batch_size=1)
    mask = np.squeeze(mask)
    mask = (mask > 0.5).astype(np.uint8)
    subframes_mask.append(mask)

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

def highest_pixel_brightness_ignore_black(image_matrix):
    mask = np.any(image_matrix != [0, 0, 0], axis=-1)
    non_black_pixels = image_matrix[mask][:, 0]
    if len(non_black_pixels) > 0:
        return np.max(non_black_pixels)
    else:
        return 0

def extract_temperatures_and_time(frame_path, reader):
    frame_img = cv2.imread(frame_path)
    if frame_img is None or frame_img.size == 0:
        print(f"Invalid frame image at {frame_path}")
        return None, None, None
    height, width = frame_img.shape[:2]
    print(f"Frame image dimensions: height={height}, width={width}")
    roi_horario = frame_img[10:60, 30:290]
    roi_temp_superior = frame_img[0:200, 700:900]
    roi_temp_inferior = frame_img[500:600, 700:810]
    if roi_horario.size == 0:
        print("ROI 'roi_horario' is empty.")
        return None, None, None
    if roi_temp_superior.size == 0:
        print("ROI 'roi_temp_superior' is empty.")
        return None, None, None
    if roi_temp_inferior.size == 0:
        print("ROI 'roi_temp_inferior' is empty.")
        return None, None, None
    reader = easyocr.Reader(['en'])
    result_horario = reader.readtext(roi_horario)
    result_temp_superior = reader.readtext(roi_temp_superior)
    result_temp_inferior = reader.readtext(roi_temp_inferior)
    horario_text = ""
    temp_superior_text = ""
    temp_inferior_text = ""
    for detection in result_horario:
        horario_text += detection[1]
    for detection in result_temp_superior:
        temp_superior_text += detection[1]
    for detection in result_temp_inferior:
        temp_inferior_text += detection[1]
    print(f"Extracted date/time text: {horario_text}")
    print(f"Extracted temp_max text: {temp_superior_text}")
    print(f"Extracted temp_min text: {temp_inferior_text}")
    temp_max = format_and_convert_to_float(temp_superior_text)
    temp_min = format_and_convert_to_float(temp_inferior_text)
    if temp_max is None or temp_min is None:
        print("Failed to extract temperatures.")
        temp_max, temp_min = None, None
    return horario_text, temp_max, temp_min

frame_image_directory = '/content/frames'

assert len(subframes_list) == len(image_files), "Mismatch between subframes_list and image_files"

data_list = []

for idx in range(len(subframes_list)):
    subframe_image = subframes_list[idx]
    subframe_filename = image_files[idx]
    mask = subframes_mask[idx]
    if mask.shape[:2] != subframe_image.shape[:2]:
        print(f"Mask and subframe image sizes do not match for {subframe_filename}")
        continue
    if mask.dtype != bool:
        mask = mask > 0
    mask_expanded = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_subframe = np.where(mask_expanded, subframe_image, 0)
    brightest_pixel = highest_pixel_brightness_ignore_black(masked_subframe)
    frame_info_match = re.search(r'cropped_(\d+)_(\d+)_(\d+)\.png', os.path.basename(subframe_filename))
    if frame_info_match:
        frame_number = int(frame_info_match.group(1))
        x_coord = frame_info_match.group(2)
        y_coord = frame_info_match.group(3)
    else:
        print(f"Could not extract frame number from {subframe_filename}")
        continue
    frame_image_filename = f'frame_{frame_number}.png'
    frame_image_path = os.path.join(frame_image_directory, frame_image_filename)
    if not os.path.exists(frame_image_path):
        print(f"Frame image file {frame_image_path} does not exist")
        continue
    date_time_text, temp_max, temp_min = extract_temperatures_and_time(frame_image_path, reader)
    if temp_max is None or temp_min is None:
        print(f"Failed to extract temperatures for frame {frame_number}")
        continue
    temperature = temp_min + (temp_max - temp_min) * (brightest_pixel / 255.0)
    data_list.append({
        'index': idx + 1,
        'frame': frame_number,
        'date_time': date_time_text,
        'temperature': temperature
    })

df = pd.DataFrame(data_list)

excel_filename = 'output2.xlsx'
df.to_excel(excel_filename, index=False)

print(f"Excel file '{excel_filename}' has been created successfully.")
