import numpy as np
from skimage.transform import resize
import cv2

def detect(img , loaded_model):
    img = resize(img , (64 , 64))
    img = np.reshape(img , (1 , 64 , 64 , 3))
    return loaded_model.predict(np.array(img))





