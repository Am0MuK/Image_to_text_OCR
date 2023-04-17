# Import the libraries
import cv2 # For image processing
import os # For file operations
from werkzeug.utils import secure_filename  # For secure file uploads
import pytesseract # For OCR
import numpy as np # For array manipulation
import pandas as pd # For data frame manipulation
from flask import Flask, render_template, request, redirect, url_for # For web app
from pdf2image import convert_from_path # For converting PDF to images


# Define the constants
UPLOAD_FOLDER ='C:\\Projects\\Image_to_text_OCR\\uploads' # Define a folder name for uploads # Define a folder name for uploads
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'} # Add PDF to the allowed extensions
LANG = "ron+rus" # Use both English and Russian languages
PSM = 6
OEM = 1
WHITELIST = "AĂăÂâBbCcDdEeFfGgHhIiÎîJjKkLlMmNnOoPpQqRrSsȘșTtȚțUuVvWwXxYyZz1234567890.-"
BLACKLIST = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.-"

app = Flask(__name__) # Create a flask app instance
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # Set the upload folder in the app config

def allowed_file(filename):
    """Check if a file has an allowed extension

    Args:
        filename (str): The name of the file

    Returns:
        bool: True if the file has an allowed extension, False otherwise
    """
    # Get the file extension by splitting the filename at the last dot
    file_extension = filename.rsplit('.', 1)[1].lower()

    # Return True if the file extension is in the set of allowed extensions
    return file_extension in ALLOWED_EXTENSIONS

def preprocess_image(img):
    """Preprocess an image for OCR

    Args:
        img (numpy.ndarray): The input image in BGR format

    Returns:
        numpy.ndarray: The preprocessed image in binary format
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Apply an adaptive threshold to binarize the image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Create a rectangular kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # Apply a closing operation to fill small gaps and holes
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Return the preprocessed image
    return morph

def detect_text_region(img):
    """Detect the region of the image containing text

    Args:
        img (numpy.ndarray): The input image in BGR format

    Returns:
        tuple: The coordinates and size of the text region as (left, top, width, height)
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour



    # Get the bounding box of the contour
    left, top, width, height = cv2.boundingRect(max_contour)

    # Expand the bounding box slightly to include any text that may be close to the edges
    margin = 10 # Define a margin for expansion
    left = max(0, left - margin)
    top = max(0, top - margin)
    width = min(img.shape[1] - left, width + 2 * margin)
    height = min(img.shape[0] - top, height + 2 * margin)

    return left, top, width, height

def extract_text(img):
    """Extract text from an image

    Args:
        img (numpy.ndarray): The input image in BGR format

    Returns:
        pandas.DataFrame: A data frame with the extracted text as a column
    """
    # Preprocess the image
    img = preprocess_image(img)

    # Detect the text region in the image
    left, top, width, height = detect_text_region(img)

    # Crop the image to the text region
    roi = img[top:top+height, left:left+width]

    # Define the config string for pytesseract
    config = f"--psm {PSM} --oem {OEM} -c tessedit_char_whitelist={WHITELIST}"

    # Use tesseract to extract the text from the region of interest
    text = pytesseract.image_to_string(roi, lang=LANG, config=config)

    # Create a data frame with the text as a column
    df = pd.DataFrame([text], columns=["text"])

    # Return the data frame
    return df

@app.route('/') # Define the route for the home page
def home():
    return render_template('index.html') # Render the index.html template
@app.route('/upload', methods=['GET', 'POST'])

def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return 'File successfully uploaded'

    return '''
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
    '''

@app.route('/url')
def get_url():
    url = request.url
    return f'The URL of this page is {url}'

@app.route('/extract_text', methods=['GET', 'POST']) # Define the route for extracting text from an image or a PDF file
def extract_text_route():
    # If the request is a GET method
    if request.method == 'GET':
        # Render a template with a form to upload a file
        return render_template('upload.html')
    # If the request is a POST method
    elif request.method == 'POST':
        # Get the file from the request
        file = request.files.get('file')

        # Check if there is no file or if it has an invalid extension
        if not file or not allowed_file(file.filename):
            return redirect(request.url) # Redirect to the same page

        # Get the file extension by splitting the filename at the last dot
        file_extension = file.filename.rsplit('.', 1)[1].lower()

        # If the file is an image
        if file_extension in {'jpg', 'jpeg', 'png'}:
            # Decode the image file from bytes to numpy array
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            # Extract text from the image and get a data frame
            df = extract_text(img)

            # Define the output file name
            output_file = 'output_file.xlsx'

            # Try to read the existing output file
            try:
                existing_df = pd.read_excel(output_file) # Read the existing data frame from the output file
                new_df = pd.concat([existing_df, df], ignore_index=True) # Concatenate the existing and new data frames
                new_df.to_excel(output_file, index=False) # Save the new data frame to the output file
            # If output file does not exist or cannot be read, create a new one
            except:
                df.to_excel(output_file, index=False) # Save the new data frame to a new output file

            return redirect(url_for('home')) # Redirect to the home page

        # If the file is a PDF
        elif file_extension == 'pdf':
            # Convert the PDF file to a list of images using pdf2image library
            images = convert_from_path(file)

            # Loop over each image in the list
            for img in images:
                # Convert the PIL image to a numpy array
                img = np.array(img)

                # Extract text from the image and get a data frame
                df = extract_text(img)

                # Define the output file name
                output_file = 'output_file.xlsx'

                # Try to read the existing output file
                try:
                    existing_df = pd.read_excel(output_file) # Read the existing data frame from the output file
                    new_df = pd.concat([existing_df, df], ignore_index=True) # Concatenate the existing and new data frames
                    new_df.to_excel(output_file, index=False) # Save the new data frame to the output file
                # If output file does not exist or cannot be read, create a new one
                except:
                    df.to_excel(output_file, index=False) # Save the new data frame to a new output file

            return redirect(url_for('home')) # Redirect to the home page

        # If the file is neither an image nor a PDF
        else:
            return redirect(request.url) # Redirect to the same page


if __name__ == '__main__':
    app.run(debug=True) # Run the flask app in debug mode