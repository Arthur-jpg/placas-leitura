import cv2
import pytesseract

# Windows - Mude o caminho conforme necessário
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extrair_texto(placa):
    texto = pytesseract.image_to_string(placa, config='--psm 8')
    return texto.strip()

# Teste
if __name__ == "__main__":
    img = cv2.imread("data/images/teste_placa.jpg")
    texto = extrair_texto(img)
    print(f"Placa identificada: {texto}")
