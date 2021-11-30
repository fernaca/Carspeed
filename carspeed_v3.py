import cv2
import numpy as np
import imutils
import time

def obtener_centro(x, y, ancho, altura):
    centrox = x + (ancho // 2)
    centroy = y + (altura // 2)
    return centrox, centroy

cap = cv2.VideoCapture('cars2.m4v')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() # Toma el fondo y subtrae lo que seestá se moviendo
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) #crea una matriz de 3x3 con forma de elipse 
# Especificamos los puntos extremos del área a analizar
area_pts = np.array([[310,200], [380,200], [430,270], [320,270]])

dist = 10 #en metros
car_counter = 0
tim1 = 0
detec = []

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
#    area_pts = np.array([[330, 216], [frame.shape[1]-80, 216], [frame.shape[1]-80, 271], [330, 271]])

    cv2.drawContours(frame, [area_pts], -1, (255, 0, 255), 2) #Dibujo del cuadrado del area

# Con ayuda de una imagen auxiliar, determinamos el área
# sobre la cual actuará el detector de movimiento
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)   #Creamos una imagen del tamanio del frame con Negros
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)   #Dibujamos el area en blanco
    image_area = cv2.bitwise_and(frame, frame, mask=imAux)   

    # Obtendremos la imagen binaria donde la región en blanco representa
    # la existencia de movimiento
#    fgmask = fgbg.apply(frame)  En lugar de aplicarlo al frame, lo segmentamos al area
    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=8) #Control de la dilatacion para hacer figuras mas enteras
    
# Encontramos los contornos presentes de fgmask, para luego basándonos
# en su área poder determinar si existe movimiento (autos)
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(cnts) == 0:
        tim1 = 0 # Si no hay contornos, inicializar tim1 en caso que no se haya hecho antes
    for cnt in cnts:
        if cv2.contourArea(cnt) > 1500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 1)
            centro = obtener_centro(x, y, w, h) #Calculamos el centro de masa para dibujarlo luego
            detec.append(centro) #Agregamos el centro a la matriz
            cv2.circle(frame, centro, 4, (0, 0, 255), -1) #Dibujamos un punto rojo en el centro de masa

        # Si el auto vha cruzado entre 440 y 460 abierto, se incrementará
        # en 1 el contador de autos
            if 199 < y  < 205 and tim1 == 0:
                tim1= time.time() #Initial time
                print("Entró el auto: " + str(tim1))
                car_counter = car_counter + 1
                cv2.line(frame, (310, 200), (380, 200), (0, 255, 0), 3) #Linea de entrada verde

            
            if 239 < y  < 250:
                tim2 = time.time() #Final time
                print("Salió el auto: " + str(tim2))
                cv2.line(frame, (320, 270), (430, 270), (0, 255, 0), 3) #Linea de salida verde

                try:
                    velpromo = 0.0
                    velprom = (dist/((tim2-tim1)))*3.6
                    print("Velocidad (km/h) es: " + str(int(velprom)))    # + str(int(dist/((tim2-tim1))))
                    tim1 = 0
                except:
                    continue
    # Visualización del conteo de autos
    cv2.rectangle(frame, (frame.shape[1]-70, 215), (frame.shape[1]-5, 270), (0, 255, 0), 2)
    cv2.putText(frame, str(car_counter), (frame.shape[1]-55, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('Mascara', fgmask)
    
    k = cv2.waitKey(30) & 0xFF #ESC
    if k ==27:
        break
cap.release()
cv2.destroyAllWindows()