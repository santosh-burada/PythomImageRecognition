import cv2
import numpy as np
WidthHeightTarget = 320
cap = cv2.VideoCapture(0)
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)

modelConfiguration = "E:\PycharmProjects\/attendence\Yolo-Obj\/320\yolov3-320.cfg"
modelWeights = "E:\PycharmProjects\/attendence\Yolo-Obj\/320\yolov3.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
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
    print("indices",indices)
    print(len(indices))

    for i in indices:
        print(i,"i")
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (WidthHeightTarget, WidthHeightTarget), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    print(findObjects(outputs,img))
    cv2.imshow('Image', img)
    cv2.waitKey(1)
