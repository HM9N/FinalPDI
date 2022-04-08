# FinalPDI
Proyecto Final del curso de Procesamiento Digital de Imágenes 


Instrucciones:

1-) Instalar Python (Verificar que en la instalación esté incluido pip).

2-) Instalar los paquetes necesarios (Se recomienda crear un ambiente virtual de Python, aunque no es necesario). Para instalar MediaPipe ejecutar el comando: pip install mediapipe. Esto nos instalará además de MediaPipe, OpenCV y Numpy. Para instalar imutils ejecutar el comando: pip install imutils.

3-) Ejecutar el programa con su IDE de preferencia o desde consola (ejecutando el archivo index.py).

Notas:

- Poner el path dónde se encuentran las dos carpetas del dataset (withMask y withoutMask) en el archivo "dataManager.py" en el atributo "self.dataPath"
- Para entrenar el modelo, se debe ejecutar desde la linea de comandos pasando como argumento la palabra "train"
- Si ya se tiene el modelo entrenado, solo es ejecutar el programa sin argumentos de linea de comando
