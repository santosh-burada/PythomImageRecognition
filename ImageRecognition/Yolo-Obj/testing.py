import cv2
import numpy as np

WidthHeightTarget = 320
cap = cv2.VideoCapture(0)
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = "E:\PycharmProjects\/attendence\Yolo-Obj\/320\yolov3-320.cfg"
modelWeights = "E:\PycharmProjects\/attendence\Yolo-Obj\/320\yolov3.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img, c):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            # higher the confidence value means it identifies correctly.
            if confidence > 0.6:
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - w / 2), int((detection[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    # According to my testings greater the nms_threshold more overlap bounding boxes appear
    # which is good to detect the mobile phones if they overlap with faces.
    indices = cv2.dnn.NMSBoxes(bbox, confs, 0.6, nms_threshold=0.6)

    for i in indices:
        # print(i,"i")
        print(c)
        i = i[0]
        print(classNames[classIds[i]])
        if (classNames[classIds[i]]) == "cell phone":
            cv2.imwrite("E:\PycharmProjects\/attendence\Yolo-Obj\Fraud\Fraud.png"+str(c)+".png", img)
            print("Written")

c=0
while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (WidthHeightTarget, WidthHeightTarget), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    c+=1
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img, c)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
