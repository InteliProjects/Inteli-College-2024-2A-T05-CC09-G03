import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import cv2
import numpy as np

# Diretórios das subpastas de frames e os respectivos arquivos XML
frame_subfolders = [
    {'folder': 'subpasta_1', 'xml': 'annotations_1.xml', 'video_code': '00000000199000400'},
    {'folder': 'subpasta_2', 'xml': 'annotations_2.xml', 'video_code': '00000000199000400'},
    {'folder': 'subpasta_3', 'xml': 'annotations_3.xml', 'video_code': '00000000199000400'},
    {'folder': 'subpasta_4', 'xml': 'annotations_4.xml', 'video_code': '00000000199000400'},
    {'folder': 'subpasta_5', 'xml': 'annotations_5.xml', 'video_code': '00000000205000000'}
]

# Diretório onde os resultados serão salvos
output_base_folder = 'output_images'
if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)
    print(f"Pasta {output_base_folder} criada com sucesso.")

# Offset inicial para cada subpasta (ajuste conforme necessário)
frame_offsets = [0, 1282, 2564,3846, 5353]

# Itera sobre cada subpasta de frames e seu respectivo XML
for idx, subfolder_info in enumerate(frame_subfolders):
    frames_folder = subfolder_info['folder']
    xml_file = subfolder_info['xml']
    video_code = subfolder_info['video_code']
    
    # Verifica se a subpasta e o arquivo XML existem
    if not os.path.exists(frames_folder) or not os.path.exists(xml_file):
        print(f"Pasta {frames_folder} ou arquivo {xml_file} não encontrado. Pulando.")
        continue
    
    # Define o offset para a subpasta atual
    frame_offset = frame_offsets[idx]
    
    # Define o diretório de saída para a subpasta atual
    output_folder = os.path.join(output_base_folder, f'saida_subpasta_{idx + 1}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Pasta {output_folder} criada com sucesso.")

    # Parseia o arquivo XML correspondente
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Percorre todas as imagens descritas no XML
    for image_info in root.findall('.//image'):
        image_id = int(image_info.get('id'))
        frame_number = frame_offset + image_id  # Começa em 0000
        image_name = f"frame_{frame_number:04d}"  # Formato de 4 dígitos, sem extensão

        # Verifica se a imagem existe na pasta
        image_path = os.path.join(frames_folder, f"{image_name}.png")
        print(f"Tentando acessar o arquivo: {image_path}")  # Para depuração
        if not os.path.exists(image_path):
            print(f"Imagem {image_name}.png não encontrada na pasta {frames_folder}. Pulando para a próxima.")
            continue

        # Carrega a imagem
        image = Image.open(image_path)

        # Contador para múltiplos contextos
        contexto_counter = 1

        # Percorre as caixas delimitadoras no XML
        for box in image_info.findall('box'):
            label = box.get('label')

            if label == "contexto":
                contexto_coords = (
                    float(box.get('xtl')),
                    float(box.get('ytl')),
                    float(box.get('xbr')),
                    float(box.get('ybr'))
                )
                cropped_image = image.crop(contexto_coords)
                cropped_image_path = os.path.join(output_folder, f"30-07-24_{video_code}_{image_name}_contexto_{contexto_counter}_x.png")
                cropped_image.save(cropped_image_path)
                print(f"Contexto {contexto_counter} foi salvo.")

                # Preparar para desenhar os elementos dentro do contexto
                contexto_width = int(contexto_coords[2] - contexto_coords[0])
                contexto_height = int(contexto_coords[3] - contexto_coords[1])
                result_image = Image.new('RGB', (contexto_width, contexto_height), (0, 0, 0))  # 'RGB' modo para cores

                draw = ImageDraw.Draw(result_image)

                # Percorrer novamente as caixas para desenhar as cabeças primeiro
                for inner_box in image_info.findall('box'):
                    inner_label = inner_box.get('label')

                    if inner_label == "cabeca":
                        cabeca_coords = (
                            float(inner_box.get('xtl')),
                            float(inner_box.get('ytl')),
                            float(inner_box.get('xbr')),
                            float(inner_box.get('ybr'))
                        )
                        relative_cabeca_coords = (
                            cabeca_coords[0] - contexto_coords[0],
                            cabeca_coords[1] - contexto_coords[1],
                            cabeca_coords[2] - contexto_coords[0],
                            cabeca_coords[3] - contexto_coords[1]
                        )
                        draw.rectangle(relative_cabeca_coords, fill=(255, 255, 255))

                # Depois, desenhar os olhos para garantir que eles fiquem por cima
                for inner_box in image_info.findall('box'):
                    inner_label = inner_box.get('label')

                    if (inner_label == "olho"):
                        olho_coords = (
                            float(inner_box.get('xtl')),
                            float(inner_box.get('ytl')),
                            float(inner_box.get('xbr')),
                            float(inner_box.get('ybr'))
                        )
                        relative_olho_coords = (
                            olho_coords[0] - contexto_coords[0],
                            olho_coords[1] - contexto_coords[1],
                            olho_coords[2] - contexto_coords[0],
                            olho_coords[3] - contexto_coords[1]
                        )
                        draw.rectangle(relative_olho_coords, fill=(255, 0, 0))

                result_image_path = os.path.join(output_folder, f"30-07-24_{video_code}_{image_name}_contexto_{contexto_counter}_y.png")
                result_image.save(result_image_path)
                print(f"Imagem result.png para contexto {contexto_counter} gerada com sucesso.")

                contexto_counter += 1
