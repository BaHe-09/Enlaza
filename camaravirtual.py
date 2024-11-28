import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyvirtualcam  # Importar pyvirtualcam para la cámara virtual

# Cargar el modelo entrenado
model = load_model("modelo_reconocimiento_senas_1_tipo2_21.h5")

# Inicializar MediaPipe y OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Mapeo de etiquetas de clases 
class_names = ["0", "1", "2", "3", "4", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", "l", "m", "n", "p", "s", "r", "t", "v", "w", "y", "5"]

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)

# Crear la cámara virtual
with pyvirtualcam.Camera(width=640, height=480, fps=30, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    print(f'Usando la cámara virtual: {cam.device}')
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se pudo capturar el video.")
            break

        # Redimensionar el frame a 640x480
        image = cv2.resize(image, (640, 480))

        # Convertir la imagen a RGB para MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con MediaPipe
        results = hands.process(image_rgb)

        # Inicializar la etiqueta predicha
        class_label = ""

        # Si se detecta una mano, procesa los puntos clave
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraer las coordenadas x, y, z de los puntos clave
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

                # Convertir los puntos clave en un array para el modelo
                keypoints = np.array(keypoints).reshape(1, -1)

                # Realizar la predicción
                prediction = model.predict(keypoints)
                class_index = np.argmax(prediction)
                class_label = class_names[class_index]  # Obtener la clase predicha

        # Agregar la predicción como texto en el frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Seña: {class_label}"
        cv2.putText(image, text, (10, 40), font, 1, (0, 255, 255), 2)

        # Mostrar la imagen en pantalla
        cv2.imshow("Predicción en Tiempo Real", image)

        # Enviar el frame procesado a la cámara virtual
        cam.send(image)
        cam.sleep_until_next_frame()

        # Salir al presionar 'Esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
