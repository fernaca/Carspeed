import cv2
import time
# En esta version hacemos un recorte de la region para tomar menos autos

cap = cv2.VideoCapture('cars2.m4v') #('cars.mp4')  #Path to footage
car_cascade = cv2.CascadeClassifier('cars.xml')  #Path to cars.xml

#Coordinates of polygon in frame::: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
#       Linea de arriba   /  Linea de abajo
#coord=[[230,200],[320,200],[200,270],[350,270]]
x1 = 300
y1 = 170
x2 = 310
y2 = 240
coord=[[x1,y1],[430,170],[x2,y2],[490,240]]

#Distance between two horizontal lines in (meter)
dist = 10
entro_auto = False
while True:
    ret, img = cap.read()
# Marcamos la ROI Region de interes
    ROI = img[y1:y2, x1:490] #FILAS / COLUMNAS   
    gray = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
#    gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    cars=car_cascade.detectMultiScale( gray, 1.1, 1) #(gray,1.8,2)
#    cv2.imshow('ROI',ROI)
    
    for (x,y,w,h) in cars:
        #Dibujamos el rectangulo en el auto
        cv2.rectangle(img,(x+x1,y+y1),(x+w+x1,y+h+y1),(225,0,0),2)

    cv2.line(img, (coord[0][0],coord[0][1]),(coord[1][0],coord[1][1]),(0,0,255),2)   #First horizontal line
    cv2.line(img, (coord[0][0],coord[0][1]), (coord[2][0],coord[2][1]), (0, 0, 255), 2) #Vertical left line
    cv2.line(img, (coord[2][0],coord[2][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2) #Second horizontal line
    cv2.line(img, (coord[1][0],coord[1][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2) #Vertical right line

    for (x, y, w, h) in cars:

        if(x+x1>=coord[0][0] and y+y1==coord[0][1]) and entro_auto == False:
#            cv2.rectangle(gray,(0,0),(gray.shape[1],40),(0,0,0),-1)
            cv2.line(img, (coord[0][0], coord[0][1]), (coord[1][0], coord[1][1]), (0, 255,0), 2) #Changes line color to green
            tim1= time.time() #Initial time
            print("Entró el auto: " + str(tim1))
            entro_auto = True

        if (x+x2>=coord[2][0] and y+220>=coord[2][1]):
            cv2.line(img, (coord[2][0],coord[2][1]), (coord[3][0], coord[3][1]), (0, 0, 255), 2) #Changes line color to green
            tim2 = time.time() #Final time
            print("Salió el auto: " + str(tim2))
            entro_auto = False
            #We know that distance is 3m
            try:
                print("Velocidad (km/h) es: ", (dist/((tim2-tim1))*3,6))
            except:
                continue

    cv2.imshow('Imagen',img) #Shows the frame



    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()