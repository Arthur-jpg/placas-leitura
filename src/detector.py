from ultralytics import YOLO
import cv2

class PlacaDetector:
    def __init__(self, model_path="runs/detect/train5/weights/best.pt"):
        """Inicializa o detector de placas com YOLO."""
        self.model = YOLO(model_path)

    def detectar_placas(self, frame):
        """Executa a detecção de placas em um frame."""
        results = self.model(frame)
        return results[0].plot()  # Retorna a imagem com as detecções desenhadas
