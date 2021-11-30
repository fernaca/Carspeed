import cv2
import time
import numpy as np
import imutils

# En esta version hacemos un recorte de la region para tomar menos autos
cap = cv2.VideoCapture('cars2.m4v') #('cars.mp4')  #Path to footage

# Algoritmo de sustraccion
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# Para mejorar imagen binaria
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Dibujamos un rectángulo en frame, para señalar el estado
    # del área en análisis (movimiento detectado o no detectado)
    cv2.rectangle(img,(0,0),(img.shape[1],40),(0,0,0),-1)
    color = (0, 255, 0)
    texto_estado = "Estado: No hay movimientos"
    # Especificamos los puntos extremos del área a analizar
#    area_pts = np.array([[320,200], [380,200], [460,img.shape[0]], [380,img.shape[0]]])
    area_pts = np.array([[310,200], [380,200], [430,270], [320,270]])

    # Con ayuda de una imagen auxiliar, determinamos el área
    # sobre la cual actuará el detector de movimiento
    imAux = np.zeros(shape=(img.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask=imAux)
     
    # Obtendremos la imagen binaria donde la región en blanco representa
    # la existencia de movimiento
    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

# Encontramos los contornos presentes en fgmask, para luego basándonos
# en su área poder determina si existe movimiento
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0), 2)
            texto_estado = "Estado: Movimiento Detectado!"
            color = (0, 0, 255)  
    # Visuzalizamos el alrededor del área que vamos a analizar
    # y el estado de la detección de movimiento        
    cv2.drawContours(img, [area_pts], -1, color, 2)
    cv2.putText(img, texto_estado , (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)
#    cv2.imshow('fgmask', fgmask)   #Imagen binaria
    cv2.imshow("frame", img)
#    k = cv2.waitKey(70) & 0xFF
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()