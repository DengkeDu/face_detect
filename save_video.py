import numpy as np
import cv2

#face_cascade = cv2.CascadeClassifier('/home/wrsadmin/opencv/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('/home/wrsadmin/opencv/opencv-3.4.0/data/haarcascades/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('/home/wrsadmin/workdir/oopencv/face_detect/haarcascade_frontalface_default.xml')

#cap = cv2.VideoCapture("rtspsrc location=rtsp://127.0.0.1:8554/test latency=0 ! rtph264depay ! decodebin ! videoconvert ! appsink")
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    out.write(img)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
