import cv2
import os
import numpy as np


class Trainer:
    def __init__(self):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        #self.dataPath = "C:/Users/XA0L/Documents/U/2021-2/Procesamiento Digital de Imagenes/Proyecto/Entregable 1/archive/Face Mask Dataset/Train"
        self.dataPath = "C:/Users/XA0L/Documents/U/2021-2/Procesamiento Digital de Imagenes/Proyecto/Dataset_faces"
        self.labels = []
        self.facesData = []

    def train(self):
        print("Entrenando...")
        self.model.train(self.facesData, np.array(self.labels))
        self.saveModel()

    def prepareData(self):
        dir_list = os.listdir(self.dataPath)
        print("Lista archivos:", dir_list)
        label = 0
        for name_dir in dir_list:
            dir_path = self.dataPath + "/" + name_dir

            for file_name in os.listdir(dir_path):
                image_path = dir_path + "/" + file_name
                print("Leyendo imagenes...")
                image = cv2.imread(image_path, 0)
                self.facesData.append(image)
                self.labels.append(label)
            label += 1

    def saveModel(self):
        self.model.write("model.xml")
        print("El modelo ha sido almacenado")