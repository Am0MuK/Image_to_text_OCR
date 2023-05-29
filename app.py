# Import the required modules
from flask import Flask, render_template, request, redirect, url_for, flash, current_app, send_file
import os
from preprocessing import preprocess_image
from ocr import perform_ocr
from spell_correction import correct_spelling
from pdf_utils import extract_text_from_pdf
import xml.etree.ElementTree as ET
from pathlib import Path

# Define the upload folder path
UPLOAD_FOLDER = 'C:\\Users\\edyk7\\Projects\\Image_to_text_OCR\\uploads'

# Create the Flask app instance
app = Flask(__name__)
app.secret_key = "secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a helper function to check if a file has a valid extension
def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif', 'pdf'}

# Define the home route that renders the index.html template
@app.route('/')
def home():
    return render_template('index.html')

# Define the upload image route that handles the file upload and processing logic
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Get the file object from the request
    file = request.files.get('image', None)
    # If no file is found, flash a message and redirect to the home page
    if not file:
        flash('No file part')
        return redirect(request.url)

    # Get the filename from the file object
    filename = file.filename
    # If the filename is not None, proceed with the processing logic
    if filename is not None:
        # If the filename is empty, flash a message and redirect to the home page
        if filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        # If the file has a valid extension, proceed with the processing logic
        if file and is_allowed_file(filename):
            # Save the file to the upload folder path
            image_path = Path(current_app.root_path) / app.config['UPLOAD_FOLDER'] / filename
            file.save(image_path)
            # If the file is a PDF, extract text from it using pdf_utils module
            if filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(image_path)
            # Otherwise, preprocess the image using preprocessing module and perform OCR using ocr module
            else:
                preprocessed_image = preprocess_image(image_path)
                text = perform_ocr(preprocessed_image, lang='ron')

            # Correct spelling errors using spell_correction module if text is not None else use an empty string
            corrected_text = correct_spelling(text) if text is not None else ''

            # Remove empty lines from the corrected text
            corrected_text = '\n'.join([line for line in corrected_text.splitlines() if line.strip()])

            # Save the corrected text as an XML file using xml.etree.ElementTree module
            root = ET.Element('data')
            record = ET.SubElement(root, 'record')
            record.text = corrected_text
            tree = ET.ElementTree(root)

            xml_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{filename}.xml')
            xml_filepath_bytes = xml_filepath.encode() # Convert to bytes

            tree.write(xml_filepath_bytes, encoding='utf-8', xml_declaration=True)

            # Send the XML file to the user as an attachment with a download name of data.xml
            return send_file(xml_filepath, mimetype="text/xml", as_attachment=True, download_name="data.xml")
        # Otherwise, flash a message and redirect to the home page
        else:
            flash('Invalid file format. Only JPG, JPEG, PNG, GIF, and PDF files are allowed.')
            return redirect(url_for('home'))
    # If the filename is None, flash a message and redirect to the home page
    else:
        flash('No filename found')
        return redirect(url_for('home'))

# Run the app in debug mode if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True)
