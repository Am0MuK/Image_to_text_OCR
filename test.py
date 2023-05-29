# Import the required modules
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pytesseract

# Define a function to preprocess an image for OCR
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

# Define a function to perform OCR on an image
def perform_ocr(image, lang):
    psm = 3  # The page segmentation mode
    oem = 4  # The OCR engine mode
    blacklist = 'ЀЁЄІЇАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯѐёєіїабвгдежзийклмнопрстуфхцчшщъыьэюя'  # The characters to ignore
    config = f'--psm {psm} --oem {oem} -c tessedit_char_blacklist={blacklist}'  # The configuration string

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Set the path to tesseract.exe
    text = pytesseract.image_to_string(image, lang='ron', config=config)
    print(text)  # Check if text is not empty
    return text
# Define a list of test images and their ground truth labels
test_images = ['test1.jpg', 'test2.jpg', 'test3.jpg']
test_labels = ['Hello world', 'This is a test', 'OCR is fun']

# Initialize empty lists to store the predictions and scores
predictions = []
scores = []

# Loop through each test image
for image_path in test_images:
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Perform OCR and get the predicted text
    predicted_text = perform_ocr(preprocessed_image)

    # Append the predicted text to the predictions list
    predictions.append(predicted_text)

# Loop through each test label and prediction pair
for label, prediction in zip(test_labels, predictions):
    # Calculate the accuracy score for the pair
    accuracy = accuracy_score(label, prediction)

    # Calculate the precision score for the pair
    precision = precision_score(label, prediction)

    # Calculate the recall score for the pair
    recall = recall_score(label, prediction)

    # Calculate the F1 score for the pair
    f1 = f1_score(label, prediction)

    # Append the scores to the scores list as a tuple
    scores.append((accuracy, precision, recall, f1))

# Print the predictions and scores for each test image
for i in range(len(test_images)):
    print(f'Image: {test_images[i]}')
    print(f'Label: {test_labels[i]}')
    print(f'Prediction: {predictions[i]}')
    print(f'Accuracy: {scores[i][0]}')
    print(f'Precision: {scores[i][1]}')
    print(f'Recall: {scores[i][2]}')
    print(f'F1-score: {scores[i][3]}')
