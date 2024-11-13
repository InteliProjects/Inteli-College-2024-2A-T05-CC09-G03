import streamlit as st
import pandas as pd
import cv2
import tempfile
import easyocr
import numpy as np
import os
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import re

# Configuração da página
st.set_page_config(
    page_title="Medidor de Temperatura do Boi",
    page_icon="🐮",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Função para carregar o CSS do arquivo
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Carregar o CSS personalizado
load_css("style.css")

# Título e descrição
st.title("Medidor de Temperatura do Boi 🐮")
st.markdown("""
Esta aplicação permite medir a temperatura de bovinos através de uma câmera termográfica.
Desenvolvido pelo grupo **Bulleyes**.
""")

# Definição da função de perda personalizada
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

# Carregar os modelos pré-treinados
@st.cache_resource
def load_models():
    # Carregar o modelo YOLO
    model_yolo = YOLO('modelos/best.pt')
    # Carregar o modelo Keras
    model_segmentation = load_model('modelos/my_model.keras',
                                    custom_objects={'dice_loss': dice_loss})
    return model_yolo, model_segmentation

model_yolo, model_segmentation = load_models()

# Funções auxiliares
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

def extract_temperatures_and_time(frame, reader):
    # Implementação da função para extrair data/hora e temperaturas via OCR
    height, width = frame.shape[:2]
    roi_horario = frame[10:60, 30:290]
    roi_temp_superior = frame[0:200, 700:900]
    roi_temp_inferior = frame[500:600, 700:810]
    result_horario = reader.readtext(roi_horario)
    result_temp_superior = reader.readtext(roi_temp_superior)
    result_temp_inferior = reader.readtext(roi_temp_inferior)
    horario_text = ''.join([detection[1] for detection in result_horario])
    temp_superior_text = ''.join([detection[1] for detection in result_temp_superior])
    temp_inferior_text = ''.join([detection[1] for detection in result_temp_inferior])
    temp_max = format_and_convert_to_float(temp_superior_text)
    temp_min = format_and_convert_to_float(temp_inferior_text)
    return horario_text, temp_max, temp_min

# Upload do vídeo
uploaded_video = st.file_uploader("Faça upload do seu vídeo", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    st.text("Processando vídeo...")
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Ler o vídeo
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_in_seconds = total_frames / fps
    delta_seconds = 1.0  # Intervalo entre frames (ajuste conforme necessário)
    timestamps = np.arange(0, duration_in_seconds, delta_seconds)
    frames = []

    # Diretório temporário para salvar frames e imagens cortadas
    frame_output_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    # Inicializar o leitor EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)

    # Ler e salvar frames do vídeo
    for idx, t in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_filename = os.path.join(frame_output_dir, f'frame_{idx}.png')
        cv2.imwrite(frame_filename, frame)
    cap.release()

    # Listas para armazenar resultados
    data_list = []

    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Processar cada frame
    for idx, frame in enumerate(frames):
        # Atualizar barra de progresso
        progress = (idx + 1) / len(frames)
        progress_bar.progress(progress)
        status_text.text(f"Processando frame {idx + 1}/{len(frames)}")

        # Previsão com o modelo YOLO
        results = model_yolo.predict(frame)
        if not results:
            continue

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped_image = frame[y1:y2, x1:x2]
                if cropped_image.size == 0:
                    continue
                # Pré-processar a imagem para o modelo de segmentação
                image_resized = cv2.resize(cropped_image, (128, 128))
                image_array = img_to_array(image_resized) / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                # Previsão com o modelo de segmentação
                mask = model_segmentation.predict(image_array, batch_size=1)
                mask = np.squeeze(mask)
                mask = (mask > 0.5).astype(np.uint8)
                mask = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]))

                # Aplicar a máscara na imagem cortada
                mask_expanded = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                masked_subframe = np.where(mask_expanded, cropped_image, 0)

                # Obter o pixel mais brilhante ignorando o preto
                brightest_pixel = highest_pixel_brightness_ignore_black(masked_subframe)

                # Extrair data/hora e temperaturas do frame via OCR
                date_time_text, temp_max, temp_min = extract_temperatures_and_time(frame, reader)

                if temp_max is None or temp_min is None:
                    continue

                # Calcular a temperatura
                temperature = temp_min + (temp_max - temp_min) * (brightest_pixel / 255.0)

                # Adicionar os dados à lista
                data_list.append({
                    'Index': len(data_list) + 1,
                    'Frame': idx,
                    'Date/Time': date_time_text,
                    'Temperature': temperature
                })

    # Remover a barra de progresso e a mensagem de status
    progress_bar.empty()
    status_text.empty()

    if len(data_list) == 0:
        st.error("Não foi possível extrair dados do vídeo. Verifique se o vídeo está correto e tente novamente.")
    else:
        # Criar um DataFrame a partir dos dados
        df = pd.DataFrame(data_list)

        # Gerar o arquivo Excel em um arquivo temporário
        excel_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        df.to_excel(excel_file.name, index=False)

        # Fornecer o link para download
        with open(excel_file.name, 'rb') as f:
            excel_data = f.read()
        st.download_button(label='Baixar resultado em Excel', data=excel_data, file_name='temperature_readings.xlsx')

        st.success("Processamento concluído!")

# Footer
footer = """
<div class="footer">
    &copy; 2023 Bulleyes. Todos os direitos reservados.
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
