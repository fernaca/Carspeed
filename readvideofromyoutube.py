import streamlink
import cv2

url = 'https://www.youtube.com/watch?v=bZ6NL59FMoc'

streams = streamlink.streams(url)
print(streams)
#cap = cv2.VideoCapture(streams["360p"].url)
cap = cv2.VideoCapture(streams["best"].url)

while True:
   ret, frame = cap.read()
print(frame)
cv2.imshow('frame', frame)

if cv2.waitKey(1) & 0xff == ord('q'):
    breakpointq

cap.release()
cv2.destroyAllWindows()