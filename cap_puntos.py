import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# Inicializar MediaPipe y OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inicializar lista para almacenar los puntos clave y la etiqueta de clase
data = []

# Nombre del archivo CSV donde se guardarán los datos
output_csv = "puntos_clave1.csv"

# Solicitar al usuario la etiqueta de clase (ej., "A", "1", etc.)
label = input("Ingresa la clase (letra o número) para esta captura: ")

print("Presiona 's' para guardar los puntos clave en el CSV, 'c' para cambiar la clase, o 'Esc' para salir.")

# Iniciar captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("No se pudo capturar el video.")
        break

    # Convertir la imagen de BGR a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe
    results = hands.process(image_rgb)

    # Dibujar puntos clave y conexiones en la mano detectada
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer las coordenadas x, y, z de cada punto clave
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            # Mostrar la imagen con los puntos clave
            cv2.imshow('MediaPipe Hands', image)

    # Capturar la tecla presionada
    key = cv2.waitKey(1) & 0xFF

    # Guardar los puntos clave si se presiona 's'
    if key == ord('s') and results.multi_hand_landmarks:
        data.append(keypoints + [label])
        print(f"Puntos clave guardados para la clase '{label}'.")

    # Cambiar la clase si se presiona 'c'
    elif key == ord('c'):
        label = input("Ingresa la nueva clase (letra o número) para esta captura: ")
        print(f"Clase cambiada a '{label}'.")

    # Salir del bucle al presionar 'Esc'
    elif key == 27:
        break

# Cerrar la captura de video
cap.release()
cv2.destroyAllWindows()

# Guardar los datos en un archivo CSV, agregando si ya existe
if os.path.exists(output_csv):
    # Si el archivo ya existe, cargamos el contenido para agregarle los nuevos datos
    existing_data = pd.read_csv(output_csv)
    # Crear DataFrame de los nuevos datos con las mismas columnas
    data_df = pd.DataFrame(data, columns=existing_data.columns)
    # Agregar nuevos datos al archivo existente
    data_df.to_csv(output_csv, mode='a', header=False, index=False)
else:
    # Si el archivo no existe, crear uno nuevo con las columnas iniciales
    columns = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)] + ["label"]
    data_df = pd.DataFrame(data, columns=columns)
    # Guardar el archivo CSV por primera vez con el encabezado
    data_df.to_csv(output_csv, index=False)

print(f"Datos guardados en {output_csv}.")
