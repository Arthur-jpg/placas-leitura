import cv2
from src.detector import PlacaDetector

def processar_video(video_path, output_path="data/videos/carros.mp4"):
    """
    Processa um vídeo aplicando a detecção de placas.
    """
    detector = PlacaDetector("runs/detect/train5/weights/best.pt")  # Inicializa o detector

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Aplica a detecção de placas
        frame = detector.detectar_placas(frame)

        cv2.imshow("Detecção de Placas", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✅ Processamento concluído! Vídeo salvo em: {output_path}")
