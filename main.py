import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
from keras.preprocessing.image import img_to_array, array_to_img
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Define dataset path
dataset_path = "D:\Project Copy\Basic Machine Learning Projects\Plant Disease Detection\Dataset"
labels = os.listdir(dataset_path)
print(labels)

# Function to convert images to arrays
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, (256, 256))
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error: {e}")
        return None

# Load dataset
root_dir = os.listdir(dataset_path)
image_list, label_list = [], []
all_labels = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
binary_labels = [0, 1, 2]

for temp, directory in enumerate(root_dir):
    plant_image_list = os.listdir(f"{dataset_path}/{directory}")
    for file in plant_image_list:
        image_path = f"{dataset_path}/{directory}/{file}"
        image_list.append(convert_image_to_array(image_path))
        label_list.append(binary_labels[temp])

# Convert lists to NumPy arrays
label_list = np.array(label_list)
image_list = np.array(image_list, dtype=np.float16) / 255.0

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state=10)

# Reshape input images
x_train = x_train.reshape(-1, 256, 256, 3)
x_test = x_test.reshape(-1, 256, 256, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), padding="same", input_shape=(256, 256, 3), activation="relu"),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D(16, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax")
])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

# Split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=10)

# Train the model
epochs = 50
batch_size = 128
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

# Save the model
model.save("plant_disease_model.h5")

# Plot training history
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], color='r', label='train')
plt.plot(history.history['val_accuracy'], color='b', label='val')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Evaluate model
print("Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1] * 100:.2f}%")

# Prediction
y_pred = model.predict(x_test)
img = array_to_img(x_test[11])

print("Originally:", all_labels[np.argmax(y_test[11])])
print("Predicted:", all_labels[np.argmax(y_pred[11])])

# Display predictions for first 50 images
for i in range(50):
    print(all_labels[np.argmax(y_test[i])], " - ", all_labels[np.argmax(y_pred[i])])
