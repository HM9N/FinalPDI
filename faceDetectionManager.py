import cv2
import mediapipe as mp
import imutils


class FaceDetectionManager:

    def __init__(self, videoManager):
        self.mp_face_detection = mp.solutions.face_detection # Solución de MediaPipe para detección de rostros
        self.mp_drawing = mp.solutions.drawing_utils # Para poder visualizar rectangulo que rodea el rostro y los 6 puntos claves
        self.videoManager = videoManager # Se usa la clase VideoManager
        self.labels=["WithMask","WithoutMask"]

    # El método tiene toda la lógica para hacer la detección de rostros
    def detectFace(self):

        # Se inicia la captura por medio de la camara
        cap = self.videoManager.beginVideoCapture()

        # Leer modelo

        # Se empieza a hacer la detección
        with self.mp_face_detection.FaceDetection(
                min_detection_confidence=0.5) as face_detection: # Min_detection_confidence: el valor minimo de confianza para que una detección sea considerada exitosa
            model = cv2.face.LBPHFaceRecognizer_create() # Se instancia el modelo para trabajar LBPH
            model.read("model.xml") # Se lee el modelo entrenado
            while True:
                ret, frame = cap.read() # Se lee el vídeo streaming
                if ret == False: 
                    break
                frame = cv2.flip(frame, 1) # Se voltea la imagen
                height, width, _ = frame.shape # Se guardan las dimensiones de la imagen
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Como openCV por defecto lee las imagenes en BGR, se transforman a RGB
                results = face_detection.process(frame_rgb) #El resultado de la detección 
                if results.detections is not None:
                    for detection in results.detections: # Se trabaja con resultados y coordenadas de la imagen (Y los puntos que trabaja MediaPipe)
                        xmin = int(detection.location_data.relative_bounding_box.xmin * width) 
                        ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                        w = int(detection.location_data.relative_bounding_box.width * width)
                        h = int(detection.location_data.relative_bounding_box.height * height)
                        if xmin < 0 and ymin < 0: # En ocasiones daba valores negativos y nos generaba errores, por lo que controlamos esos errores
                         continue
                        face_image = frame[ymin : ymin + h, xmin : xmin + w] # Se extrae el rostro para poder trabajar con el dataset
                        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) # Se transforma a escala de grises para poder trabajar con LBPH
                        face_image = cv2.resize(face_image, (72, 72), interpolation=cv2.INTER_CUBIC) # Se redimensiona el rostro a las dimensiones de las imagenes del dataset
                        result = model.predict(face_image) # Se hace la prediccón
                        if result[1] < 150: # Se cambia el color y se agrega el texto cuando se tiene mascarilla o no
                            color = (0, 255, 0) if self.labels[result[0]] == "WithMask" else (0, 0, 255)
                        cv2.putText(frame, "{}".format(self.labels[result[0]]), (xmin, ymin - 15), 2, 1, color, 1, cv2.LINE_AA) # Se agrega el texto
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2) 
        
                cv2.imshow("Frame", frame)
                k = cv2.waitKey(1)
                if k == 27:
                 break   
        cap.release()
        cv2.destroyAllWindows()        