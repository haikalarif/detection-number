from unicodedata import name
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views
# from .views import deteksi_plat_nomor, hasil_deteksi
from .views import upload_image, webcam_detection

urlpatterns = [
    path('', upload_image, name='upload_image'),
    path('real_time/', webcam_detection, name='webcam_detection'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
