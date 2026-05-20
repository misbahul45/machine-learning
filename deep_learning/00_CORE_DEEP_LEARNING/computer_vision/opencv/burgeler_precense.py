import cv2
import numpy as np

# Buka webcam
cap = cv2.VideoCapture(0)

# Periksa apakah webcam berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak dapat membuka webcam. Periksa izin kamera di System Preferences.")
    exit()

print("Webcam berhasil dibuka. Tekan 'q' untuk keluar.")

# Ambil frame pertama
ret, frame1 = cap.read()
if not ret:
    print("Error: Tidak dapat membaca frame pertama.")
    cap.release()
    exit()

ret, frame2 = cap.read()
if not ret:
    print("Error: Tidak dapat membaca frame kedua.")
    cap.release()
    exit()

while cap.isOpened():
    # Hitung perbedaan antar frame
    diff = cv2.absdiff(frame1, frame2)

    # Convert ke grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Blur supaya noise berkurang
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilasi biar area gerak lebih jelas
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Cari contour
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Gambar kotak dan teks untuk manusia yang terdeteksi
    human_count = 0
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        human_count += 1

    # Tampilkan jumlah manusia yang terdeteksi
    if human_count == 0:
        cv2.putText(frame1, "Nothing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame1, "1 Human", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Tampilkan frame asli dan frame dengan deteksi gerak
    cv2.imshow("Original", frame1)
    cv2.imshow("Motion Detection", frame1)

    # Geser frame
    frame1 = frame2
    ret, frame2 = cap.read()

    # Keluar tekan q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()