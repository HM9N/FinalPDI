import cv2
import os
import numpy as np


class Trainer:
    def __init__(self):
        self.model = cv2.face.LBPHFaceRecognizer_create() #Se instancia el modelo para entremiento y predicción (LBPH)
        self.dataPath = "C:/Users/XA0L/Documents/U/2021-2/Procesamiento Digital de Imagenes/Proyecto/Dataset_faces" # Se ingresa la ruta del dataset
        self.labels = [] # Etiquetas asociadas a las imagenes 
        self.facesData = [] # Arreglo donde se almacenarán los rostros

    def train(self):
        print("Entrenando...")
        self.model.train(self.facesData, np.array(self.labels))
        self.saveModel()

    def prepareData(self):
        dir_list = os.listdir(self.dataPath) # Se listan las carpetas del dataset
        label = 0 # variable que tomará valor de 0 o 1
        for name_dir in dir_list:
            dir_path = self.dataPath + "/" + name_dir # El path de cada imagen del dataset a trabajar 

            for file_name in os.listdir(dir_path):
                image_path = dir_path + "/" + file_name  #Se leen las imagenes
                print("Leyendo imagenes...")
                image = cv2.imread(image_path, 0)
                self.facesData.append(image) # Se agregan los rostros al arreglo facesdata
                self.labels.append(label) # Se agregan las etiquetas
            label += 1

    def saveModel(self):
        self.model.write("model.xml") # Se guarda el modelo resultante del entrenamiento
        print("El modelo ha sido almacenado")