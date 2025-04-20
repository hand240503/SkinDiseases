# views.py
import os
import shutil
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import ImageUploadForm
from .models import SkinDiseaseModel  # Import class từ models.py

# Khởi tạo model AI một lần
model_instance = SkinDiseaseModel()

def predict_disease(request):
    result = None
    image_url = None

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']

        fs = FileSystemStorage()
        temp_filename = fs.save(image_file.name, image_file)
        temp_path = os.path.join(settings.MEDIA_ROOT, temp_filename)

        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        new_path = os.path.join(upload_dir, temp_filename)
        shutil.move(temp_path, new_path)

        # Gọi hàm dự đoán từ class
        result = model_instance.predict(new_path)

        image_url = os.path.join(settings.MEDIA_URL, 'uploads', temp_filename)

    return render(request, 'predictor/index.html', {
        'form': ImageUploadForm(),
        'result': result,
        'image_url': image_url
    })
