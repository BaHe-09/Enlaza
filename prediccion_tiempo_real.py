import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model("modelo_reconocimiento_senas_1_tipo2_21.h5")

# Inicializar MediaPipe y OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Mapeo de etiquetas de clases 
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u","o","b", "c","d", "f","g","h","l","m","n","p","s", "r","t","v","w","y"]

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

    # Configuración de texto y posición
    image_height, image_width, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text = f"- {class_label} -"

    # Obtener tamaño del texto para calcular la posición centrada
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = (image_width - text_width) // 2  # Calcular posición x centrada
    text_y = image_height - 15  # Posición y en la parte inferior

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