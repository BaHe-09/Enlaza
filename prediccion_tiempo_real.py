import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

# Cargar el modelo entrenado
model = load_model("modelo_reconocimiento_senas_1_tipo2_21.h5")

# Inicializar MediaPipe y OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Mapeo de etiquetas de clases
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o"]

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)

# Variables para deletrear palabras
current_word = ""  # Palabra que se está formando
last_letter = ""  # Última letra detectada para evitar duplicados consecutivos
frame_counter = 0  # Contador de fotogramas para estabilizar la letra detectada
<<<<<<< Updated upstream
stabilization_frames = 10  # Número de fotogramas consecutivos para estabilizar una letra
=======
stabilization_frames = 7  # Número de fotogramas consecutivos para estabilizar una letra
>>>>>>> Stashed changes

# Variables para limpiar la palabra
last_detection_time = time.time()  # Último momento en que se detectó una letra
timeout = 5  # Tiempo en segundos sin detección para borrar la palabra

# Configuración para pantalla completa
<<<<<<< Updated upstream
cv2.namedWindow("Predicción en Tiempo Real", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Predicción en Tiempo Real", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
=======
#cv2.namedWindow("Predicción en Tiempo Real", cv2.WINDOW_NORMAL)
#cv2.setWindowProperty("Predicción en Tiempo Real", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
>>>>>>> Stashed changes


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("No se pudo capturar el video.")
        break

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe
    results = hands.process(image_rgb)

    # Procesar puntos clave
    class_label = ""  # Predicción de la seña
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extraer las coordenadas x, y, z de cada punto clave
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            # Convertir a numpy y redimensionar para el modelo
            keypoints = np.array(keypoints).reshape(1, -1)  # Convertir a formato de entrada del modelo

            # Realizar la predicción
            prediction = model.predict(keypoints)
            class_index = np.argmax(prediction)
            class_label = class_names[class_index]  # Obtener la clase predicha

            # Si la letra es igual a la última detectada, incrementar el contador
            if class_label == last_letter:
                frame_counter += 1
            else:
                frame_counter = 0  # Reiniciar el contador si cambia la letra
                last_letter = class_label  # Actualizar la última letra

            # Si se estabiliza por los fotogramas requeridos, añadir la letra
            if frame_counter >= stabilization_frames:
                current_word += class_label
                frame_counter = 0  # Reiniciar el contador después de añadir la letra

            # Actualizar el tiempo de la última detección
            last_detection_time = time.time()

    # Borrar la palabra si no hay detección durante el tiempo especificado
    if time.time() - last_detection_time > timeout:
        current_word = ""

    # Configuración de texto y posición
    image_height, image_width, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    # Mostrar la letra detectada en tiempo real
    text_letter = f"Letra: {class_label}"
    (text_width, text_height), _ = cv2.getTextSize(text_letter, font, font_scale, font_thickness)
    text_x = (image_width - text_width) // 2  # Posición centrada
    text_y = 50  # Parte superior
    cv2.putText(image, text_letter, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

    # Mostrar la palabra formada
    text_word = f"Palabra: {current_word}"
    (text_width, text_height), _ = cv2.getTextSize(text_word, font, font_scale, font_thickness)
    text_x = (image_width - text_width) // 2  # Posición centrada
    text_y = image_height - 50  # Parte inferior
    cv2.putText(image, text_word, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)

    # Mostrar la imagen en pantalla
    cv2.imshow("Predicción en Tiempo Real", image)

    # Salir al presionar 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cerrar la captura de video
cap.release()
cv2.destroyAllWindows()
