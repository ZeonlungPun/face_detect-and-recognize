import cv2,pickle
import numpy as np

labels={'names':1}
with open("labels.pickle","rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

fascer=cv2.CascadeClassifier('E:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_recognizer.yml')



frame=cv2.imread('E:\opencv\\tracy35.jpg')
frame=cv2.resize(frame,(500,500))
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces = fascer.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)
for (x,y,w,h) in faces:
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=frame[y:y+h,x:x+w]
    id_,conf=recognizer.predict(roi_gray)
    print('predict:',labels[id_],'conf:',conf)
    cv2.putText(frame,labels[id_],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0))
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


cv2.imshow('frame',frame)
cv2.waitKey(0)


