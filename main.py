from tensorflow.keras.models import load_model
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import os 

model = load_model('modelo_2.h5')

def preproccess_image(image_path):
    image_p = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_p = cv2.bitwise_not(image_p)
    resized_image = cv2.resize(image_p, (48,48))
    resized_image[np.where(resized_image > 190)] = 255
    resized_image[np.where(resized_image < 125)] = 0

    normalized_image = resized_image / 255.0 
    processed_image = np.expand_dims(normalized_image, axis = -1)
    # processed_image = np.expand_dims(processed_image, axis=0)

    return processed_image, resized_image

# image_p = './images/seis_p.png'
# image_pp, image_sz  = preproccess_image(image_p)
# prediction = model.predict(image_pp)
# prediction_label = np.argmax(prediction)
# plt.imshow(image_sz, cmap='gray')
# plt.title(f'Label Predict: {prediction_label}')
# plt.axis('off')
# plt.show()

files_path = './images_2'
files = os.listdir(files_path)

images = []
images_prep = []

for file in files:
    image_p, image = preproccess_image(os.path.join(files_path, file))
    images.append(image_p)
    images_prep.append(image)

images = np.array(images)

predictions = model.predict(images)
predictions_labels = np.argmax(predictions, axis = 1)

num_rows = 2
num_columns = 7 
plt.figure(figsize=(num_columns*2, num_rows*2))

for i in range(len(images_prep)):
    img = images_prep[i]
    plt.subplot(num_rows, num_columns, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted label {predictions_labels[i]}', fontsize = 8)
    plt.axis('off')
plt.tight_layout()
plt.show()





