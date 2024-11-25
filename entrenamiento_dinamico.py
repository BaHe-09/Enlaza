from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el dataset
data = pd.read_csv("puntos_clave_dinamicos.csv")

# Separar características (X) y etiquetas (y)
X = data.drop("label", axis=1).values.reshape(-1, 30, 63)  # 30 frames, 63 puntos clave (21*3)
y = pd.factorize(data["label"])[0]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo LSTM
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30, 63), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),

    LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(len(np.unique(y)), activation='softmax')  # Salidas igual al número de etiquetas únicas
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)

# Entrenar el modelo
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    validation_data=(X_test, y_test), 
                    batch_size=32,
                    callbacks=[early_stopping, reduce_lr])

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")

# Guardar el modelo
model.save("modelo_reconocimiento_gestos_dinamicos.h5")
print("Modelo guardado")
