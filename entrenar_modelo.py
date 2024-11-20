import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Cargar el dataset de puntos clave
data = pd.read_csv("puntos_clave_tipo2.csv")

# Separar características (X) y etiquetas (y)
X = data.drop("label", axis=1).values  # Todas las columnas excepto la etiqueta
y = pd.factorize(data["label"])[0]     # Convertir etiquetas a índices numéricos

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo de red neuronal optimizado
model = Sequential([
    Dense(256, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(36, activation='softmax')  # 36 salidas para 26 letras + 10 números
])

# Compilar el modelo con un optimizador ajustado
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks de EarlyStopping y reducción de tasa de aprendizaje
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)

# Entrenar el modelo con el callback de early stopping y reduce_lr
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    validation_data=(X_test, y_test), 
                    batch_size=64,
                    callbacks=[early_stopping, reduce_lr])

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")

# Guardar el modelo entrenado
model.save("modelo_reconocimiento_senas_1_tipo2_21.h5")
print("Modelo guardado")
