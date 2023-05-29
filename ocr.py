import pytesseract

def perform_ocr(image, lang):
    psm = 3  # The page segmentation mode
    oem = 4  # The OCR engine mode
    blacklist = 'ЀЁЄІЇАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯѐёєіїабвгдежзийклмнопрстуфхцчшщъыьэюя'  # The characters to ignore
    config = f'--psm {psm} --oem {oem} -c tessedit_char_blacklist={blacklist}'  # The configuration string

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Set the path to tesseract.exe
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    print(text)  # Check if text is not empty
    return text
