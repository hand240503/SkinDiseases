import os
import shutil
import tensorflow as tf
import numpy as np
from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load model
model_path = r'D:\VKU\ky_6\Machine_Learning(2)\output\trained_skin_disease_model.keras'
model = tf.keras.models.load_model(model_path)

class_names = [
    '1. Eczema 1677',
    '2. Melanoma 15.75k',
    '3. Atopic Dermatitis - 1.25k',
    '4. Basal Cell Carcinoma (BCC) 3323',
    '5. Melanocytic Nevi (NV) - 7970'
]

def predict_disease(request):
    result = None
    image_url = None

    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']

        # Lưu file tạm thời vào MEDIA_ROOT
        fs = FileSystemStorage()
        temp_filename = fs.save(image_file.name, image_file)
        temp_path = os.path.join(settings.MEDIA_ROOT, temp_filename)

        # Tạo thư mục uploads nếu chưa có
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        # Tạo đường dẫn mới cho ảnh
        new_path = os.path.join(upload_dir, temp_filename)

        # Di chuyển ảnh từ file tạm vào thư mục uploads
        shutil.move(temp_path, new_path)

        # Dự đoán
        img = load_img(new_path, target_size=(128, 128))
        input_arr = img_to_array(img)
        input_arr = np.expand_dims(input_arr, axis=0)

        predictions = model.predict(input_arr)
        result_index = np.argmax(predictions)
        model_prediction = class_names[result_index]
        result = model_prediction

        # Đường dẫn ảnh để hiển thị
        image_url = os.path.join(settings.MEDIA_URL, 'uploads', temp_filename)

    return render(request, 'predictor/index.html', {
        'form': ImageUploadForm(),
        'result': result,
        'image_url': image_url
    })
