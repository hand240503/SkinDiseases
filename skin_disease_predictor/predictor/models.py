# models.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class SkinDiseaseModel:
    def __init__(self):
        model_path = r'D:\VKU\ky_6\Machine_Learning(2)\output\trained_skin_disease_model.keras'
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = [
            '1. Eczema 1677',
            '2. Melanoma 15.75k',
            '3. Atopic Dermatitis - 1.25k',
            '4. Basal Cell Carcinoma (BCC) 3323',
            '5. Melanocytic Nevi (NV) - 7970'
        ]

    def predict(self, image_path):
        img = load_img(image_path, target_size=(128, 128))
        input_arr = img_to_array(img)
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = self.model.predict(input_arr)
        result_index = np.argmax(predictions)
        return self.class_names[result_index]
