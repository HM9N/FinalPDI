import cv2

class VideoManager:

    def __init__(self):
        pass

    # Se inicia la captura del vídeo por medio de la cámara 
    def beginVideoCapture(self):
        print("Voy a ejecutar")
        return cv2.VideoCapture(0, cv2.CAP_DSHOW) # Se realiza la captura del video