import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocessing steps
    size = (1000, 1000) # Define a size variable as a tuple of two integers
    image = cv2.resize(image, size)  # Resize using the size variable
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\resized.jpg', image) # Save the resized image
    image = cv2.medianBlur(image, 5)  # Blur
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\blurred.jpg', image) # Save the blurred image
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Binarize
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\binarized.jpg', image) # Save the binarized image
    image = cv2.bitwise_not(image)  # Invert
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\inverted.jpg', image) # Save the inverted image
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))  # Open
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\opened.jpg', image) # Save the opened image
    image = cv2.dilate(image, None, iterations=3)  # Dilate
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\dilated.jpg', image) # Save the dilated image
    image = cv2.erode(image, None, iterations=2)  # Remove noise
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\eroded.jpg', image) # Save the eroded image

    # De-skew
    coords = cv2.findNonZero(image)  # Find non-zero pixels
    rect = cv2.minAreaRect(coords)  # Find minimum area rect
    angle = rect[-1]  # The rotation angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Get the rotation matrix
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\deskewed.jpg', image) # Save the deskewed image

    # Find the contours using cv2.findContours()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask image of the same shape as the original image
    mask = np.zeros_like(image)

    # Draw the contours on the mask image using cv2.drawContours()
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\\contoured.jpg', mask) # Save the contoured image

    # Filter out the contours that are too small or too large
    area_thresh = 100  # You can adjust this value as needed
    filtered_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            filtered_contours.append(c)

    # Create a new mask image with only the filtered contours
    new_mask = np.zeros_like(image)
    cv2.drawContours(new_mask, filtered_contours, -1, (255, 255, 255), -1)
    cv2.imwrite('C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\temp\filtered.jpg', new_mask) # Save the filtered image

    # Apply the mask to the original image using cv2.bitwise_and()
    segmented = cv2.bitwise_and(image, new_mask)
    
    return segmented

