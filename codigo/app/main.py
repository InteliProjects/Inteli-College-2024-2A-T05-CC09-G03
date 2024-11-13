import streamlit as st
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
import tempfile
import os
from moviepy.editor import VideoFileClip

# Título e descrição
st.title("Medidor de Temperatura do Boi 🐮")
st.markdown("""
Esta aplicação permite medir a temperatura de bovinos através de vídeos coloridos e térmicos.
Faça upload dos seus vídeos, selecione o intervalo de tempo para análise e processe os resultados.
""")

# Função para definir a perda Dice
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice

# Função para carregar os modelos
@st.cache_resource
def load_models():
    # Ajuste os caminhos para seus modelos
    model_path = '../deteccao_cabeca/modelo_deteccao_cabeca_completo.pth'  # Atualize com o caminho do seu modelo
    model2_path = '../modelo_segmentacao/modelo_segmentacao.keras'  # Atualize com o caminho do seu modelo

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    model2_carregado = load_model(model2_path, custom_objects={'dice_loss': dice_loss})

    return model, device, model2_carregado

# Carregar modelos
model, device, model2_carregado = load_models()

# Inicializar o leitor OCR
reader = easyocr.Reader(['en'])

# Função para calcular IoU entre duas caixas delimitadoras
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

# Função para calcular a distância do centróide entre duas caixas delimitadoras
def compute_centroid_distance(box1, box2):
    centroid1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    centroid2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    distance = np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
    return distance

# Função para formatar e converter o texto OCR em float
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

# Função para extrair temperaturas e timestamp do frame usando OCR
def extract_temperatures_from_frame(frame, reader):
    # O frame deve ter 640x512 após o redimensionamento

    # Coordenadas fixas para temp_max e temp_min
    roi_temp_max = frame[50:120, 500:575]
    roi_temp_min = frame[400:440, 500:575]

    # Coordenadas para extração do timestamp
    roi_timestamp = frame[10:50, 200:350]

    # Verificar se as ROIs são válidas
    if roi_temp_max.size == 0 or roi_temp_min.size == 0 or roi_timestamp.size == 0:
        print("ROI para extração de temperatura ou timestamp está vazia.")
        return None, None, None

    # Pré-processamento das ROIs para melhor precisão do OCR
    roi_temp_max_gray = cv2.cvtColor(roi_temp_max, cv2.COLOR_BGR2GRAY)
    roi_temp_min_gray = cv2.cvtColor(roi_temp_min, cv2.COLOR_BGR2GRAY)

    _, roi_temp_max_thresh = cv2.threshold(roi_temp_max_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, roi_temp_min_thresh = cv2.threshold(roi_temp_min_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    roi_timestamp_gray = cv2.cvtColor(roi_timestamp, cv2.COLOR_BGR2GRAY)
    _, roi_timestamp_thresh = cv2.threshold(roi_timestamp_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Usar as imagens pré-processadas para OCR
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

# Upload dos vídeos
color_video_file = st.file_uploader("Faça upload do vídeo colorido", type=["mp4", "avi", "mov"])
thermal_video_file = st.file_uploader("Faça upload do vídeo térmico", type=["mp4", "avi", "mov"])

if color_video_file and thermal_video_file:
    # Salvar os vídeos enviados em arquivos temporários
    color_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    color_tempfile.write(color_video_file.read())
    color_video_path = color_tempfile.name

    thermal_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    thermal_tempfile.write(thermal_video_file.read())
    thermal_video_path = thermal_tempfile.name

    # Obter as durações dos vídeos originais
    color_clip = VideoFileClip(color_video_path)
    thermal_clip = VideoFileClip(thermal_video_path)
    color_duration = color_clip.duration
    thermal_duration = thermal_clip.duration
    min_duration = min(color_duration, thermal_duration)

    # Fechar os clipes para liberar recursos
    color_clip.close()
    thermal_clip.close()

    # Selecionar intervalo de tempo
    start_time = st.number_input("Tempo inicial (segundos)", min_value=0.0, max_value=min_duration, value=0.0)
    end_time = st.number_input("Tempo final (segundos)", min_value=0.0, max_value=min_duration, value=min_duration)

    if st.button("Processar Vídeos"):
        # Verificar se o processamento já foi feito
        if 'processed' not in st.session_state:
            st.session_state.processed = False

        if not st.session_state.processed:
            # Função de processamento
            def process_videos(color_video_path, thermal_video_path, start_time, end_time):
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
                from moviepy.editor import VideoFileClip

                # Definir intervalos de recorte para os frames coloridos e térmicos
                color_crop_rows = (0, 720)
                color_crop_cols = (250, 500)

                # Inicializar captura de vídeo
                color_cap = cv2.VideoCapture(color_video_path)
                thermal_cap = cv2.VideoCapture(thermal_video_path)

                # Obter FPS original e contagens de frames
                original_fps_color = color_cap.get(cv2.CAP_PROP_FPS)
                original_fps_thermal = thermal_cap.get(cv2.CAP_PROP_FPS)

                # FPS desejado
                desired_fps = 10

                # Calcular o passo de frame
                frame_step_color = int(round(original_fps_color / desired_fps))
                frame_step_thermal = int(round(original_fps_thermal / desired_fps))

                # Obter contagens totais de frames
                total_frames_color = int(color_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_frames_thermal = int(thermal_cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Converter tempos de início e fim para números de frames
                start_frame_color = int(start_time * original_fps_color)
                end_frame_color = int(end_time * original_fps_color)
                start_frame_thermal = int(start_time * original_fps_thermal)
                end_frame_thermal = int(end_time * original_fps_thermal)

                # Garantir que ambos os vídeos tenham o mesmo número de frames após a amostragem
                total_frames = min((end_frame_color - start_frame_color) // frame_step_color, (end_frame_thermal - start_frame_thermal) // frame_step_thermal)

                # Obter tamanho do frame do vídeo colorido
                frame_width = int(color_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(color_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Escritor de vídeo de saída
                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_video = cv2.VideoWriter(output_video_path, fourcc, desired_fps, (frame_width, frame_height))

                # Definir o tamanho do bloco e calcular o número de blocos
                block_size = 50
                thermal_height = 512  # Após o redimensionamento
                thermal_width = 640   # Após o redimensionamento
                num_blocks_vert = int(np.ceil(thermal_height / block_size))
                num_blocks_hor = int(np.ceil(thermal_width / block_size))

                # Criar uma grade de blocos
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

                # Definir deslocamentos específicos para alguns blocos
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

                # Função para determinar o bloco de uma caixa delimitadora
                def get_block_of_box(box, blocks):
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2

                    for block in blocks:
                        if (block['x_start'] <= x_center < block['x_end']) and (block['y_start'] <= y_center < block['y_end']):
                            return (block['row'], block['col'])
                    return None

                # Inicializar variáveis para rastreamento
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

                # Inicializar barra de progresso
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Loop para processar os frames dos vídeos
                for idx in range(total_frames):
                    i = idx  # Índice do frame para processamento
                    frame_num_color = start_frame_color + i * frame_step_color
                    frame_num_thermal = start_frame_thermal + i * frame_step_thermal

                    # Atualizar progresso
                    progress = (idx + 1) / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processando frame {idx + 1}/{total_frames}")

                    # Definir posições dos frames
                    color_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_color)
                    thermal_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_thermal)

                    ret_color, color_frame = color_cap.read()
                    ret_thermal, thermal_frame = thermal_cap.read()

                    if not ret_color or not ret_thermal:
                        print(f"Falha ao ler frames no índice {i}")
                        break

                    # Aplicar recorte ao frame colorido
                    color_cropped = color_frame[color_crop_rows[0]:color_crop_rows[1], color_crop_cols[0]:color_crop_cols[1]]

                    # Converter o frame colorido recortado para RGB e tensor
                    img_rgb = cv2.cvtColor(color_cropped, cv2.COLOR_BGR2RGB)
                    img_tensor = T.ToTensor()(img_rgb).to(device)

                    # Detecção de cabeças no frame colorido
                    with torch.no_grad():
                        detections = model([img_tensor])

                    # Extrair caixas e scores
                    boxes = detections[0]['boxes'].cpu().numpy()
                    scores = detections[0]['scores'].cpu().numpy()

                    # Filtrar detecções por confiança
                    indices = np.where(scores > conf_threshold)[0]
                    heads = boxes[indices]

                    if len(heads) == 0:
                        print(f"Nenhuma cabeça detectada no frame {i}")
                        out_video.write(color_frame)
                        continue  # Pular o processamento deste frame

                    current_heads = []
                    current_ids = []
                    matched_ids = set()

                    # Para cada detecção atual, tentar encontrar uma correspondência com IDs existentes
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
                            max_temperatures[cow_id_counter] = None  # Inicializar como None

                    # Incrementar a idade de todos os IDs não correspondidos neste frame
                    for cid in list(cow_age.keys()):
                        if cid not in matched_ids:
                            cow_age[cid] += 1
                            if cow_age[cid] > max_age:
                                del cow_history[cid]
                                del cow_age[cid]
                                del max_temperatures[cid]

                    # Atualizar cow_ids com os IDs atuais
                    cow_ids = {cid: box for cid, box in zip(current_ids, current_heads)}

                    # Redimensionar o frame térmico para 640x512
                    thermal_resized = cv2.resize(thermal_frame, (640, 512))

                    # Extrair temp_max, temp_min e timestamp do frame térmico redimensionado
                    temp_max, temp_min, timestamp = extract_temperatures_from_frame(thermal_resized, reader)

                    if temp_max is not None and temp_min is not None and timestamp is not None:
                        print(f"Frame {i}: temp_max={temp_max}, temp_min={temp_min}, timestamp={timestamp}")
                    else:
                        print(f"Falha ao extrair temperaturas ou timestamp para o frame {i}")
                        out_video.write(color_frame)
                        continue  # Pular o processamento deste frame

                    # Preparar para desenhar anotações no frame colorido original (não recortado)
                    annotated_frame = color_frame.copy()

                    # Para cada detecção, processar o recorte do olho e anotar o frame
                    for (x_min, y_min, x_max, y_max), cid in zip(current_heads, current_ids):
                        # Mapear as coordenadas de volta para o tamanho original do frame
                        x_min_orig = int(x_min + color_crop_cols[0])
                        y_min_orig = int(y_min + color_crop_rows[0])
                        x_max_orig = int(x_max + color_crop_cols[0])
                        y_max_orig = int(y_max + color_crop_rows[0])

                        # Desenhar caixa delimitadora no frame anotado
                        cv2.rectangle(annotated_frame, (x_min_orig, y_min_orig), (x_max_orig, y_max_orig), (0, 255, 0), 2)

                        # Aplicar deslocamentos de bloco para alinhar frames coloridos e térmicos
                        block_key = get_block_of_box([x_min_orig, y_min_orig, x_max_orig, y_max_orig], blocks)
                        if block_key is not None and block_key in blocks_shifts:
                            deltaX = blocks_shifts[block_key]['deltaX']
                            deltaY = blocks_shifts[block_key]['deltaY']
                        else:
                            deltaX = 0
                            deltaY = 0

                        # Ajustar coordenadas para o frame térmico
                        x_min_th = x_min_orig + deltaX
                        y_min_th = y_min_orig + deltaY
                        x_max_th = x_max_orig + deltaX
                        y_max_th = y_max_orig + deltaY

                        # Garantir que as coordenadas estão dentro dos limites do frame
                        x_min_th = max(0, min(thermal_width - 1, x_min_th))
                        x_max_th = max(0, min(thermal_width - 1, x_max_th))
                        y_min_th = max(0, min(thermal_height - 1, y_min_th))
                        y_max_th = max(0, min(thermal_height - 1, y_max_th))

                        # Extrair a região do olho do frame térmico
                        eye_crop = thermal_resized[int(y_min_th):int(y_max_th), int(x_min_th):int(x_max_th)]
                        if eye_crop.size == 0:
                            print(f"Recorte do olho vazio para ID {cid} no frame {i}. Pulando cálculo de temperatura.")
                            continue

                        # Redimensionar e pré-processar o eye_crop
                        eye_crop_resized = cv2.resize(eye_crop, (128, 128)) / 255.0
                        eye_crop_resized = np.expand_dims(eye_crop_resized, axis=0)

                        # Realizar segmentação
                        mask = model2_carregado.predict(eye_crop_resized, batch_size=1)
                        mask = np.squeeze(mask)
                        mask = (mask > 0.5).astype(np.uint8)

                        if np.sum(mask) == 0:
                            print(f"Máscara vazia para ID {cid} no frame {i}. Pulando cálculo de temperatura.")
                            continue

                        # Calcular a intensidade máxima na área mascarada
                        gray_eye_crop_resized = cv2.cvtColor((eye_crop_resized[0] * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                        masked_eye_crop = np.where(mask == 1, gray_eye_crop_resized, 0)
                        max_intensity = np.max(masked_eye_crop)

                        if max_intensity <= 0:
                            print(f"Intensidade máxima inválida para ID {cid} no frame {i}. Pulando cálculo de temperatura.")
                            continue

                        # Converter intensidade para temperatura real
                        try:
                            temperature = temp_min + (temp_max - temp_min) * (max_intensity / 255.0)

                            # Atualizar a temperatura máxima registrada para o boi
                            if max_temperatures.get(cid) is None or temperature > max_temperatures[cid]:
                                max_temperatures[cid] = temperature
                        except Exception as e:
                            print(f"Erro ao calcular temperatura para ID {cid} no frame {i}: {e}")
                            continue

                        # Anotar o frame com ID do boi e temperatura
                        label = f'ID: {cid}, Temp: {temperature:.2f}°C'
                        cv2.putText(annotated_frame, label, (x_min_orig, y_min_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Anotar o frame com o timestamp
                    cv2.putText(annotated_frame, f'Time: {timestamp}', (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Registrar dados de rastreamento
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

                    # Escrever o frame anotado no vídeo de saída
                    out_video.write(annotated_frame)

                # Liberar recursos de vídeo
                color_cap.release()
                thermal_cap.release()
                out_video.release()

                progress_bar.empty()
                status_text.empty()

                # Re-encode o vídeo para compatibilidade com o navegador
                clip = VideoFileClip(output_video_path)
                output_video_web_path = output_video_path.replace('.mp4', '_web.mp4')
                clip.write_videofile(output_video_web_path, codec='libx264')

                # Remover o vídeo original para economizar espaço
                os.remove(output_video_path)

                return tracking_data, output_video_web_path  # Retornar o novo caminho do vídeo

            # Chamar a função de processamento
            tracking_data, output_video_path = process_videos(color_video_path, thermal_video_path, start_time, end_time)

            # Salvar os resultados no estado da sessão
            st.session_state.processed = True
            st.session_state.tracking_data = tracking_data
            st.session_state.output_video_path = output_video_path

            st.success("Processamento concluído!")

        # Exibir o vídeo anotado
        video_file = open(st.session_state.output_video_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

        # Converter os dados de rastreamento em DataFrame
        df = pd.DataFrame(st.session_state.tracking_data)

        # Exibir o DataFrame como tabela
        st.dataframe(df)

        # Botões de download
        # Para o vídeo
        st.download_button('Baixar Vídeo Anotado', video_bytes, file_name='annotated_video.mp4', mime='video/mp4')

        # Para o arquivo CSV
        csv_file = df.to_csv(index=False).encode('utf-8')
        st.download_button('Baixar Dados CSV', data=csv_file, file_name='tracking_data.csv', mime='text/csv')

    else:
        if 'processed' in st.session_state and st.session_state.processed:
            st.success("Processamento concluído!")

            # Exibir o vídeo anotado
            video_file = open(st.session_state.output_video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

            # Converter os dados de rastreamento em DataFrame
            df = pd.DataFrame(st.session_state.tracking_data)

            # Exibir o DataFrame como tabela
            st.dataframe(df)

            # Botões de download
            # Para o vídeo
            st.download_button('Baixar Vídeo Anotado', video_bytes, file_name='annotated_video.mp4', mime='video/mp4')

            # Para o arquivo CSV
            csv_file = df.to_csv(index=False).encode('utf-8')
            st.download_button('Baixar Dados CSV', data=csv_file, file_name='tracking_data.csv', mime='text/csv')
