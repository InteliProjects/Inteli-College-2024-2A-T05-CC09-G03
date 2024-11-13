# Importações necessárias
import cv2
import numpy as np
import easyocr
import re

def format_and_convert_to_float(ocr_text):
    """
    Função para formatar o texto extraído por OCR e converter para float.
    """
    cleaned_text = re.sub(r'[^\d]', '', ocr_text)
    
    if len(cleaned_text) >= 2:
        formatted_text = cleaned_text[:2] + '.' + cleaned_text[2:]
    else:
        formatted_text = cleaned_text + '.0'

    return float(formatted_text)

def highest_pixel_brightness_ignore_black(image_matrix):
    """
    Função que encontra o pixel mais brilhante, ignorando os pixels pretos.
    """
    mask = np.any(image_matrix != [0, 0, 0], axis=-1)
    non_black_pixels = image_matrix[mask][:, 0]
        
    if len(non_black_pixels) > 0:
        return np.max(non_black_pixels)
    else:
        return 0

def leitura_de_temperatura(frame, mask):
    """
    Função que lê a imagem de entrada, realiza OCR em regiões de interesse, e calcula a temperatura com base no brilho do pixel.
    """
    # Carregar a imagem do frame
    img = cv2.imread(frame)

    # Definir as coordenadas das regiões de interesse (ROI)
    roi_horario = img[int(46.67):int(114.67), int(77.73):int(726.93)]  # Área onde o horário/data está
    roi_temp_superior = img[int(136.00):int(193.33), int(1069.73):int(1220.27)]  # Área da temperatura superior
    roi_temp_inferior = img[int(845.33):int(901.33), int(1072.27):int(1216.27)]  # Área da temperatura inferior

    # Criar o leitor do EasyOCR
    reader = easyocr.Reader(['en'])

    # Realizar OCR em cada ROI
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

    print("\nTexto completo de Horário:", horario_text)
    print("Texto completo de Temperatura Máxima:", temp_superior_text)
    print("Texto completo de Temperatura Mínima:", temp_inferior_text)

    temp_max = format_and_convert_to_float(temp_superior_text)
    temp_min = format_and_convert_to_float(temp_inferior_text)

    print("Temperatura máxima para cálculo:", temp_max)
    print("Temperatura mínima para cálculo:", temp_min)

    # Carregar a máscara e converter em matriz
    temp_x_matrix = np.array(img)
    temp_y_matrix = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    # Criar uma matriz resultante
    result_matrix = np.zeros_like(temp_x_matrix)

    for i in range(temp_y_matrix.shape[0]):
        for j in range(temp_y_matrix.shape[1]):
            if temp_y_matrix[i, j] == 255:  # Se o pixel em temp_y for branco
                result_matrix[i, j] = temp_x_matrix[i, j]  # Mantém valor correspondente
            else:  # Se o pixel em temp_y for preto
                result_matrix[i, j] = [0, 0, 0]  # Mantém preto

    # Salvar a imagem resultante
    result_image = result_matrix
    cv2.imwrite('temp_result.png', result_image)

    # Obter o valor do pixel mais claro ignorando os pretos
    pixel_value = highest_pixel_brightness_ignore_black(result_matrix)
    print("O pixel mais claro tem o valor de:", pixel_value)

    # Regra de três para escalar o valor do pixel para a faixa de temperatura
    temperature = temp_min + (pixel_value / 255 * (temp_max - temp_min))
    print("A temperatura do olho é de:")

    return temperature

# Exemplo de uso da função
print(leitura_de_temperatura(
    "./imagens/frame_3289.png",
    "./imagens/maskarapredita.png"
))