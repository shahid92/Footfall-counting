from centroidtracker import CentroidTracker
import time
import imutils
import numpy as np
import cv2

direction=None
ct = CentroidTracker()
(H, W) = (None, None)
#mean_value=(78.4263377603, 87.7689143744, 114.895847746)
mean_value=(104.0, 177.0, 123.0)
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list = ['Male', 'Female']

camera =cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX

face_net = cv2.dnn.readNetFromCaffe(
        "data_models/face_deploy_ssd.prototxt",
        "data_models/face_caffe_ssd.caffemodel")

def model_hub(frame):
        global detections        
        f=frame.copy()
        blob = cv2.dnn.blobFromImage(f,1,(227,227),mean_value, swapRB=False)
        face_net.setInput(blob)
        detections = face_net.forward()
        
def file_lengthy(fname):
        with open(fname) as f:
                for i, l in enumerate(f):
                        pass
                return i+1
        
while True:
        ret, frame = camera.read()
        frame = imutils.resize(frame, width=400)
        cv2.line(frame,(270,0),(270,400),(255,0,0),5)
        #cv2.line(frame,(0,0),(511,511),(255,0,0),5)
        if W is None or H is None:
                (H, W) = frame.shape[:2]
        model_hub(frame)
        rects = []
        for i in range(0, detections.shape[2]):
                if detections[0, 0, i, 2] > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        rects.append(box.astype("int"))
                        (startX, startY, endX, endY) = box.astype("int")
                        cv2.rectangle(frame,(startX, startY), (endX, endY),(0,0,255), 2)
                        #cv2.putText(frame,overlay_text,(startX,startY),font,1,(0,0,255),1,cv2.LINE_AA)
        objects = ct.update(rects)
        for (objectID, centroid) in objects.items():
                text = "ID: {}".format(objectID)
                cv2.putText(frame, text, (centroid[0],centroid[1]),font,0.5, (0, 255, 0), 1)

                if (startX+endX)>500:
                        direction="EAST"
                        fobj = open("data_report/demo.txt")
                        text = fobj.read().strip().split()

                        
                        initial_value=file_lengthy("data_report/demo.txt")
                        if int(objectID)<int(initial_value):                                
                                continue
                        else:
                                file = open('data_report/demo.txt','a+')
                                print(str(objectID))
                                file.write("\n")
                                file.write(str(objectID))
                                file.close() 
                else:
                        direction="WEST"
                cv2.putText(frame,"LOCATION: "+str(direction),(centroid[0],centroid[1]+220),font,0.5,(0,0,255), 1)
        cv2.imshow("Frame", frame)
        key =cv2.waitKey(1) & 0xFF
        if key==ord("q"):
                break

camera.release()
cv2.destroyAllWindows()
