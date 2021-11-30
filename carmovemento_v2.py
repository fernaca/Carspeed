import cv2
import numpy as np
from time import sleep
import time

largura_min = 80  # Largura minima del rectangulo
altura_min = 80  # Altura minima del rectangulo
offset = 6  # Error permitido entre pixel

delay = 60  # FPS do vídeo
detec = []
dist = 5
velprom = 0.0
tim1 = 0.0
def obtener_centro(x, y, ancho, altura):
    """
    :param ancho: ancho del objeto
    :param altura: altura doelobjeto
    :return: tupla con coordenadas del centro del objeto
    """
    centrox = x + (ancho // 2)
    centroy = y + (altura // 2)
    return centrox, centroy

def set_info(detec):
    global carros_in, carros_out, tim1, velprom
    for (x, y) in detec:
        if tim1 == 0:
            tim1= time.time() #Initial time
        if (linea_en_area + offset) > y > (linea_en_area - offset):
            tim2 = time.time() #Final time
            carros_in += 1
            cv2.line(frame1, (0, linea_en_area), (300, linea_en_area), (0, 127, 255), 3)    
            detec.remove((x, y))#Borramos para no volver a contabilizarlo
            try:
                velprom = (dist/((tim2-tim1))) *(3.6)
                tim1 = 0
            except:
                continue

def show_info(frame1, dilatada):
    text = f'Entraron: {carros_in}'
    cv2.putText(frame1, text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    text = "Velocidad (km/h):  " + str(int(velprom))
    cv2.putText(frame1, text, (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame1)
#    cv2.imshow("Detectar", dilatada)


carros_in = 0
cap = cv2.VideoCapture('video.mp4')
substraccion = cv2.bgsegm.createBackgroundSubtractorMOG()  # Toma el fondo y subtrae lo que esta en movimiento

# Coordenadas del area
x1 = 400
y1 = 450
x2 = 600
y2 = 450
x3 = 250
y3 = 720
x4 = 550
y4 = 720

linea_original = 550
linea_en_area = y3 - y1 - 170

coord=[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
# Crea una matriz de 5x5 entre 0 e 1 con forma de elipse
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  
while True:
    ret, frame = cap.read()  # Leemos el frame
    tempo = float(1 / delay)
    sleep(tempo)  # Dá um delay entre cada processamento
    frame1 = frame[y1:y4, x1:x4] #creamos un frame con el area de interes para reducir los calculos
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Pasamos a blanco y negro
    blur = cv2.GaussianBlur(grey, (3, 3), 5)  # Aplicarmos un blur para remover imperfecciones
    img_sub = substraccion.apply(blur)  # Hacemos la subraccion de la imagen
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))  # "Engrosamos" lo que sobro de la substraccion
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)  # Encerramos agujeros de adentro
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

#    contornos, img = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Buscamos contornos en el frame
    contornos, img = cv2.findContours(dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Buscamos contornos en el frame
    cv2.line(frame, (350, linea_original), (550, linea_original), (255, 127, 0), 3) #Linea carril izquierdo

#Dibujamos el area a analizar
    cv2.line(frame, (coord[0][0],coord[0][1]),(coord[1][0],coord[1][1]),(0,0,255),2)   #First horizontal line
    cv2.line(frame, (coord[0][0],coord[0][1]), (coord[2][0],coord[2][1]), (0, 0, 255), 2) #Vertical left line
    cv2.line(frame, (coord[2][0],coord[2][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2) #Second horizontal line
    cv2.line(frame, (coord[1][0],coord[1][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2) #Vertical right line
    if len(contornos) == 0 and len(detec) != 0:
        detec.clear() #Inicialiamos detec para agilizar
        tim1 = 0
    for (i, c) in enumerate(contornos):
        (x, y, w, h) = cv2.boundingRect(c) #Tomamos coordenadas y tamanio del contorno para verificarlo
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

# Es un contorno considerado (es un auto)-> Dibujamos un rectangulo
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2) 
        centro = obtener_centro(x, y, w, h) #Calculamos el centro de masa para dibujarlo luego
        detec.append(centro) #Agregamos el centro a la matriz
        cv2.circle(frame1, centro, 4, (0, 0, 255), -1) #Dibujamos un punto rojo en el centro de masa

# De los contornos seleccionados, vemos si pasaron por las lineas
    set_info(detec)
    show_info(frame, dilatada)

    if cv2.waitKey(1) == 27:
        break

print("Autos detectados: " + str(carros_in))
cv2.destroyAllWindows()
cap.release()