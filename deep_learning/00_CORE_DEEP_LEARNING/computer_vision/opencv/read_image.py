import cv2
import os

image_path = os.path.join('data', 'foto.jpeg')


if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' tidak ditemukan. Pastikan path dan nama file benar.")
else:
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Tidak bisa membaca file '{image_path}'. Pastikan file adalah gambar yang valid (JPEG/PNG).")
    else:
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()