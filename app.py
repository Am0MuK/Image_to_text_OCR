from flask import Flask, render_template, request, redirect, url_for, flash, current_app, send_file
import os 
import cv2
import pandas as pd
import pytesseract
from pytesseract import Output
import pandas as pd
import os
import PyPDF2
import phunspell

UPLOAD_FOLDER = r'C:\Users\edyk7\Projects\Image_to_text_OCR\uploads'
LANG = 'ron' # The language code
PSM = 6 # The page segmentation mode
OEM = 4 # The OCR engine mode 
BLACKLIST = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя' # The characters to ignore
CONFIG = f'–psm {PSM} --oem {OEM} -c tessedit_char_blacklist={BLACKLIST}' # The configuration string

app = Flask(__name__)
app.secret_key = "secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif', 'pdf'}
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['image']
    filename = file.filename

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and is_allowed_file(file.filename):
        filename = file.filename
        image_path = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'], filename) 
        file.save(image_path)
        if filename.lower().endswith('.pdf'):
            # Extract text from PDF file
            pdf_reader = PyPDF2.PdfReader(open(image_path, 'rb'))
            text = ''
            for page_num in range(pdf_reader.pages.__len__()):
                page = pdf_reader.pages[page_num] 
                text += page.extract_text()

        else:
            # OCR image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Preprocessing
            image = cv2.resize(image, None, fx=2, fy=2) # Resize
            image = cv2.medianBlur(image, 5) # Blur
            image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # Binarize	
            image = cv2.bitwise_not(image) # Invert
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))) # Open
            image = cv2.dilate(image, None, iterations=3) # Dilate
            image = cv2.erode(image, None, iterations=2)   # Remove noise
        
        # De-skew
            coords = cv2.findNonZero(image) # Find non-zero pixels
            rect = cv2.minAreaRect(coords) # Find minimum area rect
            angle = rect[-1] # The rotation angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0) # Get the rotation matrix
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    	# OCR
            text = pytesseract.image_to_string(image, lang=LANG, config=CONFIG, output_type=Output.STRING)
            print(text) # check if text is not empty

        # Spell correction
            hobj=phunspell.Phunspell('ro_RO')
            words = text.split()
            corrected_words = []
            for word in words:
                if not hobj.lookup(word):
                    suggestions = hobj.suggest(word)
                    suggestion_list = list(suggestions)
                if len(suggestion_list) > 0:
                    corrected_words.append(suggestion_list[0])
            else:
                corrected_words.append(word)

            text = ' '.join(corrected_words)
            print(text) # check if text is not empty and correct


        # Remove empty lines
            text = '\n'.join(corrected_words) # join the list of words with newline characters
            text = '\n'.join([line for line in text.splitlines() if line.strip()]) # remove empty lines

        #  
            df = pd.DataFrame({'text': [text]})
            df.to_xml(os.path.join(app.config['UPLOAD_FOLDER'], filename + '.xml'), index=False)

        # Send file to user
            return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename + '.csv'), mimetype="text/csv", as_attachment=True, download_name="data.csv")


    else:
        flash('Image successfully uploaded and OCR performed')
        return redirect(url_for('home')) 
    
if __name__ == "__main__":
    app.run(debug=True)
    