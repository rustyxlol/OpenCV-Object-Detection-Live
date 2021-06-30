import cv2

#Threshold for object detection
thresh = 0.5
cap = cv2.VideoCapture(1) # Try other integers(0,1,2) if you have more than one video driver. 0 otherwise.
cap.set(3, 640) # Not necessary
cap.set(4, 480) # Not necessary

classNames = []
classFile = 'coco.names'

with open(classFile,'r') as file:
    classNames = file.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#MODEL CREATION
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, conf, bbox = net.detect(img, confThreshold=thresh)
    print(classIds, bbox)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), conf.flatten(),bbox):
            cv2.rectangle(img,box,color=(255,0,0), thickness=2)
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            cv2.putText(img, str(round(confidence*100,2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("output",img)
    cv2.waitKey(1)

