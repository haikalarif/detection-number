from django.shortcuts import render
from django.core.files.storage import default_storage
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Fungsi resize image
def resize_image(image_path, max_width=1000, max_height=600):
    # Read gambar
    image = cv2.imread(image_path)

    # Dapatkan ukuran gambar
    height, width, _ = image.shape

    # Hitung rasio skala untuk menyesuaikan ukuran gambar
    scale = min(max_width / width, max_height / height)

    # Resizing gambar
    resized_image = cv2.resize(image, (int(width * scale), int(height * scale)))

    # Simpan gambar yang telah diresize
    resized_image_path = f'media/{os.path.splitext(os.path.basename(image_path))[0]}_resized.jpg'
    cv2.imwrite(resized_image_path, resized_image)

    return resized_image_path

# Fungsi untuk mendeteksi plat nomor
def detect_license_plate(image_path):
    # Resize Image
    resized_image_path = resize_image(image_path)

    # Read Image
    image = cv2.imread(resized_image_path)

    # Ubah gambar menjadi skala abu-abu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduksi noise dengan menggunakan bilateral filter
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Reduksi noise dengan menggunakan metode GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Deteksi tepi menggunakan Canny
    edged = cv2.Canny(blurred, 50, 150)

    # Temukan kontur dalam gambar
    # contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fokus pada kontur yang memiliki aspek ratio sesuai dengan plat nomor
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        if 2.2 < aspect_ratio < 4.5:
            # Gambar kotak di sekitar plat nomor
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Potong dan ambil ROI (Region of Interest) dari gambar
            roi = gray[y:y + h, x:x + w]
            # Thresholding untuk mendapatkan teks lebih jelas
            _, threshold = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # Gunakan pytesseract untuk mendapatkan teks dari ROI
            text = pytesseract.image_to_string(threshold, config='--psm 8')
            
            if text:
                return text

    # Tampilkan gambar hasil (Anda bisa menyimpannya atau mengirimkannya sebagai respons HTTP)
    # cv2.imshow("Deteksi Plat Nomor", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Fungsi untuk menangani upload gambar dari pengguna

def save_detected_image(original_image_path, detected_text, detected_image_path):
    # Baca gambar asli
    original_image = Image.open(original_image_path)

    # Membuat gambar hasil deteksi dengan teks
    detected_image = original_image.copy()
    draw = ImageDraw.Draw(detected_image)

    # set ukuran font dan jenis font
    font_path = os.path.abspath("Roboto.ttf")
    font_size = 50
    
    font = ImageFont.truetype(font_path, font_size)

    # Koordinat dan ukuran kotak
    box_width, box_height = 1000, 500
    box_x, box_y = 20, 20

    # Gambar kotak pada plat nomor
    # draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], outline=(0, 255, 0), width=5)
    # text_position = (box_x, box_y)
    
    # Hitung posisi teks agar berada di tengah bagian bawah gambar
    text_position = ((detected_image.width - font_size) // 2, detected_image.height - font_size - 20)

    draw.text(text_position, detected_text, font=font, fill=(100, 255, 200))

    # Simpan gambar hasil deteksi
    detected_image.save(detected_image_path)

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        image_path = f'media/{uploaded_image.name}'
        
        # Simpan gambar ke media folder
        with open(image_path, 'wb') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        # Panggil fungsi deteksi plat nomor
        detected_text = detect_license_plate(image_path)

        # Simpan gambar hasil deteksi
        detected_image_path = f'media/{os.path.splitext(uploaded_image.name)[0]}_detected.jpg'
        save_detected_image(image_path, detected_text, detected_image_path)

        # Dapatkan URL gambar hasil deteksi
        detected_image = default_storage.url(detected_image_path)

        # if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        #     # Jika request menggunakan AJAX, kirim respons JSON
        #     return JsonResponse({'detected_text': detected_text, 'detected_image': detected_image})

        # return HttpResponse("Deteksi Plat Nomor Selesai")
        return render(request, 'deteksi.html', {'detected_text': detected_text, 'detected_image': detected_image})


    return render(request, 'deteksi.html')


def detect_plate_realtime():
    cap = cv2.VideoCapture(0)
    
    # Inisialisasi detektor wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Baca frame dari kamera
        ret, frame = cap.read()

        # Ubah frame menjadi skala abu-abu
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reduksi noise dengan menggunakan bilateral filter
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Loop melalui setiap wajah yang terdeteksi
        for (x, y, w, h) in faces:
            # Gambar kotak di sekitar wajah
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame, "Cakep Banget...", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
        # Reduksi noise dengan menggunakan metode GaussianBlur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
        # Deteksi tepi menggunakan Canny
        edged = cv2.Canny(blurred, 50, 150)
        # Normalisasi
        # cv::normalize($plate, $plate, 0, 255, cv::NORM_MINMAX, cv::CV_8UC1);

        # Temukan kontur dalam frame
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fokus pada kontur yang memiliki aspek ratio sesuai dengan plat nomor
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            if 2.5 < aspect_ratio < 4.0:
                # Potong dan ambil ROI (Region of Interest) dari frame
                roi = gray[y:y + h, x:x + w]
                
                # Thresholding untuk mendapatkan teks lebih jelas
                _, threshold = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                
                # Gunakan pytesseract untuk mendapatkan teks dari ROI
                text = pytesseract.image_to_string(threshold, config='--psm 8')

                # Gambar kotak di sekitar plat nomor pada frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Tampilkan teks hasil deteksi
                if text:
                    cv2.putText(frame, f"Plat Nomor: {text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tampilkan frame
        cv2.imshow("Deteksi Plat Nomor Real-Time", frame)

        # Tekan tombol 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def webcam_detection(request):
    # Panggil fungsi deteksi plat nomor secara real-time
    detect_plate_realtime()

    return render(request, 'deteksi.html')
    # return HttpResponse("Deteksi Plat Nomor Real-time")