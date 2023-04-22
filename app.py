from flask import Flask, render_template, request, redirect, url_for, flash
import os 
import cv2
import pytesseract
import pandas as pd
import os
import PyPDF2
from spellchecker import SpellChecker

UPLOAD_FOLDER = r'C:\Users\edyk7\Projects\Image_to_text_OCR\uploads'

app = Flask(__name__)
app.secret_key = "secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif', 'pdf'}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and is_allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if filename.lower().endswith('.pdf'):
            # Extract text from PDF file
            pdf_reader = PyPDF2.PdfFileReader(open(image_path, 'rb'))
            text = ''
            for page_num in range(pdf_reader.getNumPages()):
                page = pdf_reader.getPage(page_num)
                text += page.extractText()

        else:
            # OCR image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Preprocessing
        image = cv2.medianBlur(image, 5)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))
        image = cv2.dilate(image, None, iterations=3)

        # De-skew
        coords = cv2.findNonZero(image)
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # OCR
        text = pytesseract.image_to_string(image, lang='ron', config='--psm 6 --oem 4 -c tessedit_char_blacklist=АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя')

        # Spell correction
        spell = SpellChecker(language='ro')
        words = text.split()
        corrected_words = [spell.correction(word) for word in words]
        text = ' '.join(corrected_words)

        # Remove empty lines
        text = '\n'.join([line for line in text.splitlines() if line.strip()])

        # Save to CSV file
        df = pd.DataFrame({'text': [text]})
        df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename + '.csv'), index=False)

        flash('Image successfully uploaded and OCR performed')
        return redirect(url_for('home'))
    else:
        flash('Image successfully uploaded and OCR performed')
        return redirect(url_for('home'))
    
if __name__ == "__main__":
    app.run(debug=True)
