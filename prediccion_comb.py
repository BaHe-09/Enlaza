import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import pandas as pd

# Cargar los modelos
static_model = load_model("modelo_estatico.h5")  # Modelo para señas estáticas
dynamic_model = load_model("modelo_dinamico.h5")  # Modelo para gestos dinámicos

# Leer las etiquetas desde archivos CSV
static_labels = pd.read_csv("puntos_clave.csv")['label'].unique().tolist()
dynamic_labels = ["gesto1", "gesto2", "gesto3"]  # Reemplaza con las etiquetas reales del modelo dinámico

# Inicializar MediaPipe y OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Configuración para el modelo dinámico
frame_window = deque(maxlen=30)  # Ventana deslizante para almacenar puntos clave de frames

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("No se pudo capturar el video.")
        break

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe
    results = hands.process(image_rgb)

    # Inicializar etiqueta de predicción
    class_label = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar puntos clave en la mano
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer las coordenadas x, y, z de cada punto clave
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            # Para el modelo estático
            static_prediction = static_model.predict(np.array(keypoints).reshape(1, -1))
            static_class = static_labels[np.argmax(static_prediction)]

            # Para el modelo dinámico
            frame_window.append(keypoints)  # Agregar puntos clave actuales a la ventana
            if len(frame_window) == frame_window.maxlen:
                # Preparar secuencia para el modelo dinámico
                sequence = np.array(frame_window).reshape(1, frame_window.maxlen, -1)
                dynamic_prediction = dynamic_model.predict(sequence)
                dynamic_class = dynamic_labels[np.argmax(dynamic_prediction)]

                # Decidir si mostrar el gesto dinámico o la seña estática
                class_label = f"Dinámico: {dynamic_class}" if np.max(dynamic_prediction) > 0.7 else f"Estático: {static_class}"
            else:
                class_label = f"Estático: {static_class}"

    # Mostrar predicción en pantalla
    image_height, image_width, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text = f"- {class_label} -"

    # Obtener tamaño del texto para centrarlo
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = (image_width - text_width) // 2
    text_y = image_height - 15

    # Contorno negro
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2, lineType=cv2.LINE_AA)
    # Texto amarillo encima del contorno
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 255), font_thickness, lineType=cv2.LINE_AA)

    # Mostrar la imagen en pantalla
    cv2.imshow("Predicción en Tiempo Real", image)

    # Salir al presionar 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cerrar la captura de video
cap.release()
cv2.destroyAllWindows()
