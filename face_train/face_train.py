import os,cv2,pickle
import numpy as np
from PIL import Image
fascer=cv2.CascadeClassifier('E:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
base_dir=os.path.dirname(os.path.abspath(__file__))
img_dir=os.path.join(base_dir,"images")
x_train=[]
y_labels=[]
current_id=0
label_ids={}

#dirname: 除了最後一個的文件名稱  base_name: 最後一個的文件名稱

for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("jfif"):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path))
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            pil_image=Image.open(path).convert("L") #grayscale
            size=(500,500)
            pil_image=pil_image.resize(size,Image.ANTIALIAS)
            img_array=np.array(pil_image,"uint8")
            faces=fascer.detectMultiScale(img_array,scaleFactor=1.5,minNeighbors=5)
            for (x,y,w,h) in faces:
                roi=img_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle","wb") as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("face_recognizer.yml")
