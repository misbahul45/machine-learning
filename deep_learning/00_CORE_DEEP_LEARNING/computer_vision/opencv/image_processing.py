import cv2
import os

image_path = os.path.join('data', 'foto.jpeg')

if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' tidak ditemukan. Pastikan path dan nama file benar.")
else:
    os.makedirs('./data/generated', exist_ok=True)
    original_image = cv2.imread(image_path)

    # change image into gray image
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./data/generated/gray_image.jpg', gray_image)

    # change image by resizing
    rezised_image = cv2.resize(original_image, (300, 200))
    cv2.imwrite('./data/generated/resized_imag.jpg', rezised_image)

    # Define the center of the original image
    (h, w) = original_image.shape[:2]
    center = (w // 2, h // 2)
    # Rotation matrix
    matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
    # Perform the rotation on the original image
    rotated_image = cv2.warpAffine(original_image, matrix, (w, h))
    cv2.imwrite('./data/generated/rotated_image.jpg', rotated_image)

    # bluring image
    blurred_image = cv2.GaussianBlur(original_image, (15, 15), 0)
    cv2.imwrite('./data/generated/Blurred_image.jpg', blurred_image)

    # detect edge
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    cv2.imwrite('./data/generated/edge.jpg', edges)

    # drawing shape and adding text
    image_with_shapes = original_image.copy()
    cv2.rectangle(image_with_shapes, (50, 50), (200, 200), (255, 0, 0), 3)
    cv2.putText(image_with_shapes, 'trying', (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite('./data/generated/adding_shape_and_image.jpg', image_with_shapes)

    # thresholding
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, tresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('./data/generated/treshold_image.jpg', tresh_image)

    # contours detection
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    countours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite('./data/generated/contours_image.jpg', edges)