import tensorflow as tf 
import math 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import cv2

def data_bal(path):
    data = os.listdir(path)
    df = {i: [] for i in range(len(data))}
    for i in range(len(data)):
        a = f'{path}/{i}/'
        df[i] = [f"{a}{file}" for file in os.listdir(f'{path}/{i}')]
    return df

x = './new_data'
df = data_bal(x)

min_df = min(len(images) for images in df.values())
len_data = len(df)

df = {key: list(set(value))[:min_df] for key, value in df.items()}
total_images = min_df * len(df)
df = pd.DataFrame(df)

def preproccess_image(image_path):
    if os.path.exists(image_path):
        image_p = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_p = cv2.bitwise_not(image_p)
        resized_image = cv2.resize(image_p, (48,48))
        resized_image[np.where(resized_image > 190)] = 255
        resized_image[np.where(resized_image < 125)] = 0
        normalized_image = resized_image / 255.0 
        processed_image = np.expand_dims(normalized_image, axis = -1)
        return normalized_image  # Se devuelve la imagen procesada
    else:
        print(f"Warning: Unable to open or read image file: {image_path}")

y = np.repeat(df.columns, len(df))
X = []

for var in df:
    for image in df[var]:
        X.append(preproccess_image(image))
X = np.array(X)
n = 10232

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

batch_size = 32

# Definir arquitectura del modelo
model = Sequential([
    Conv2D(32, (7, 7), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')  # 10 clases en total
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience = 5)

# Entrenar el modelo usando el generador de datos de entrenamiento
history = model.fit(X_train, y_train, epochs=50,
                 validation_data=(X_test, y_test), callbacks=[early_stopping])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Mostrar un informe de clasificación
print(classification_report(y_test, predicted_labels))

model.save('modelo_2.h5')

