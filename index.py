#--------------------------------------------------------------------------
#------- PLANTILLA DE CÓDIGO ----------------------------------------------
#------- Juego PDI-------------------------------------------
#------- Por: Jhon Vásquez  y Alejandro Mercado -----------------------------------
#------- Curso Básico de Procesamiento de Imágenes y Visión Artificial-----
#------- Marzo de 2022--------------------------------------------------
#--------------------------------------------------------------------------

import faceDetectionManager # Se importa la clase FaceDetectionManager
import videoManager # Se importa la clase VideoManager
import sys 
import dataManager 

countArguments = len(sys.argv)

print(countArguments)

if(countArguments > 1 and sys.argv[1] == "train"): # Se verifica los argumentos de la linea de comandos
    print("INGRESÓ EL ARGUMENTO")
    dataManager = dataManager.Trainer()
    dataManager.prepareData()
    dataManager.train()
# Se instancia la clase VideoManager
videoManager = videoManager.VideoManager()
# Se instancia la clase FaceDetectionManager
faceDetectionManager = faceDetectionManager.FaceDetectionManager(videoManager)
# Se ejecuta el método encargado de hacer la detección de rostros
faceDetectionManager.detectFace()
