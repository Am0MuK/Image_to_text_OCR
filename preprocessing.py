from PIL import Image
import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Rescale the image to a larger size
    size = (1500, 1500)  # Define the desired size
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\resized.jpg', resized_image)

    # Convert the NumPy array to PIL Image
    pil_image = Image.fromarray(resized_image)

    # Set the DPI mode for the resized image
    dpi = (300, 300)  # Specify the DPI value
    pil_image.info['dpi'] = dpi

    # Save the resized image with DPI information
    pil_image.save('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\resized_dpi.jpg', dpi=dpi)

    # Apply contrast enhancement with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(resized_image)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\equalized.jpg', equalized_image)

    # Apply histogram matching to match the brightness and contrast of the image to a reference image
    matched_image = cv2.normalize(equalized_image, None, 0, 255, cv2.NORM_MINMAX) # type: ignore
    matched_image = cv2.LUT(matched_image, cv2.calcHist(matched_image, [0], None, [256], [0, 256])) # type: ignore
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\matched.jpg', matched_image)

    # Apply gamma correction
    gamma = 0.5
    gamma_corrected_image = np.uint8(cv2.pow(matched_image / 255.0, gamma) * 255)

    # Apply bilateral filter
    filtered_image = cv2.bilateralFilter(gamma_corrected_image, 9, 75, 75) # type: ignore
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\filtered.jpg', filtered_image)

    # Apply Otsu's thresholding
    _, thresholded_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\thresholded.jpg', thresholded_image)

    # Invert the image
    inverted_image = cv2.bitwise_not(thresholded_image)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\inverted.jpg', inverted_image)

    # De-skew the image
    coords = np.column_stack(np.where(inverted_image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = inverted_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed_image = cv2.warpAffine(inverted_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\deskewed.jpg', deskewed_image)

    # Find contours
    contours, hierarchy = cv2.findContours(deskewed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask image
    mask = np.zeros_like(deskewed_image)

    # Draw contours on the mask image
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\contoured.jpg', mask)

    # Apply the mask to the original image
    segmented_image = cv2.bitwise_and(deskewed_image, mask)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\segmented.jpg', segmented_image)

    # Apply morphological opening and closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_image = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\opened.jpg', opened_image)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\closed.jpg', closed_image)

    # Apply top-hat and black-hat transformations
    tophat_image = cv2.morphologyEx(closed_image, cv2.MORPH_TOPHAT, kernel)
    blackhat_image = cv2.morphologyEx(closed_image, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\tophat.jpg', tophat_image)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\blackhat.jpg', blackhat_image)

    # Apply Canny edge detection
    canny_image = cv2.Canny(blackhat_image, 100, 200)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\canny.jpg', canny_image)

    return canny_image
