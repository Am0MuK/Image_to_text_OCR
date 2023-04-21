from flask import Flask, render_template, request
import os
import numpy as np
import cv2
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # perform image processing and OCR here
        return 'OCR results'
    else:
        return 'No file uploaded'
    
# set the path where uploaded files will be stored
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# define the route for file upload
@app.route('/upload', methods=['POST'])
def upload():
    # get the uploaded file from the form
    file = request.files['file']
    filename = file.filename
    # save the uploaded file to the UPLOAD_FOLDER directory
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    # read the uploaded file using OpenCV
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    # rescale the image to 300 DPI
    img = cv2.resize(img, None, fx=300/img.shape[1], fy=300/img.shape[1], interpolation=cv2.INTER_CUBIC)
    
    # increase the contrast of the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # remove noise from the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    result = 255 - opening
    
    # de-skew the image
    coords = np.column_stack(np.where(result > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(result, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # set up the Tesseract OCR parameters
    config = ('-l ron --oem 4 --psm 6')
    
    # read the text from the image using Tesseract OCR
    text = pytesseract.image_to_string(result, config=config)
    
    # display the extracted text
    return render_template('result.html', text=text)

if __name__ == '__main__':
    app.run(debug=True)
