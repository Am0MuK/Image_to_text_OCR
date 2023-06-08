import pytesseract
from PIL import Image

def perform_ocr(image, lang):
    # Convert the NumPy array to a PIL Image object
    image = Image.fromarray(image)
    
    # Convert the image to RGB mode
    image = image.convert('RGB')
    
    # Use the default configuration string
    config = ''
    
    # Set the path to tesseract.exe
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Perform OCR and return the text
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    
    print(text)  # Check if text is not empty
    return text
