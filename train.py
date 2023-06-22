import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import pickle


Categories = [1, 0]
data_arr = []
target_arr = []
datadir = 'dataset/'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


for i in Categories:
    j = 0
    print(f'loading... category : {i}')
    path = os.path.join(datadir, str(i))
    for img in os.listdir(path):
        # print(j)
        try:
            img_array = imread(os.path.join(path, img))

            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            cropped_img = faces
            for (x, y, w, h) in faces:
                cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 255, 0), 4)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img_array[y:y + h, x:x + w]
                cropped_img = img_array[y:y + h, x:x + w]
            cropped_img = resize(cropped_img, (64, 64))
            # cv2.imshow("k",cropped_img)
            # cv2.waitKey(0)
            data_arr.append(cropped_img)
            target_arr.append(Categories.index(i))
        except:
            pass

        j+=1
    print(f'loaded category:{i} successfully')

X = np.array(data_arr)
y = np.array(target_arr)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2)

# Define batch size and number of epochs
batch_size = 32
epochs = 10

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Define data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

model.fit(X_train , y_train, batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val))

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)

# Print the model summary
model.summary()

print('Test loss:', score[0])
print('Test accuracy:', score[1])

with open('finalized_model.sav', 'wb') as file:
    pickle.dump(model, file)
    file.close()
