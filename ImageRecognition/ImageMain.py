import cv2
import face_recognition
import os
import numpy as np
import shutil
import pickle

print("For Search-0")
print("For load-1")

n = int(input("[0/1]"))


def ObjectDetectP1():
    global WidthHeightTarget, cap, classNames, net
    WidthHeightTarget = 320
    cap = cv2.VideoCapture(0)
    classesFile = "E:\PycharmProjects\/attendence\Yolo-Obj\coco.names"
    classNames = []
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    modelConfiguration = "E:\PycharmProjects\/attendence\Yolo-Obj\/320\yolov3-320.cfg"
    modelWeights = "E:\PycharmProjects\/attendence\Yolo-Obj\/320\yolov3.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
def ObjectDetectionP2(c):
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (WidthHeightTarget, WidthHeightTarget), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img,c)
def findObjects(outputs, img,c):
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
                cv2.imwrite("E:\PycharmProjects\/attendence\FraudFraud.png" + str(c) + ".png", img)


if n:
    os.chdir("E:\PycharmProjects\/attendence\MainImages")
    images = []
    imgNames = []
    # Now we have to go through every photo in multiple folders
    for root, dirs, files in os.walk(".", topdown=False):

        for name in files:
            imgNames.append(name.split(".")[0])
            images.append(cv2.imread(os.path.join(root, name)))
    print(imgNames, "Imagnmaes")


    # This function is used to collect the features and face location of particular image.
    def features(images):
        featuresOfImages = []
        c = 0
        print("Face Locations of Loaded Images")
        for img in images:
            imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(face_recognition.face_locations(imgs)[0])
            try:
                featuresOfImg = face_recognition.face_encodings(imgs, model='cnn')[0]
            except IndexError as e:
                print("Some Faces are not detected by dlib")
                # sys.exit(1)
            featuresOfImages.append(featuresOfImg)

        return featuresOfImages


    featuresOfTrainingImages = features(
        images)  # In this featuresOfTrainingImages list we will have the features of all
    # loaded images
    print(type(featuresOfTrainingImages))
    print("Features are collected...", len(featuresOfTrainingImages))
    with open("E:\PycharmProjects\/attendence\/featuresOfTrainingImages.txt", "wb") as fp:  # Pickling
        pickle.dump(featuresOfTrainingImages, fp)
    with open("E:\PycharmProjects\/attendence\images.txt", "wb") as fp:  # Pickling
        pickle.dump(images, fp)
    with open("E:\PycharmProjects\/attendence\imgNames.txt", "wb") as fp:  # Pickling
        pickle.dump(imgNames, fp)
else:
    ObjectDetectP1()
    c=0
    while True:
        success,img = cap.read()
        cv2.imshow("Image", img)
        c+=1
        ObjectDetectionP2(c)
        imgInput = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)


        # cv2.imshow("input image", imgInput)
        try:

            faceLocation = face_recognition.face_locations(imgInput)
            if len(faceLocation)>0:

                inputFeatures = face_recognition.face_encodings(imgInput, faceLocation, model='cnn')
                # Here we wil have the input face features
                with open("featuresOfTrainingImages.txt", "rb") as fp:  # Unpickling
                    featuresOfTrainingImages = pickle.load(fp)
                with open("imgNames.txt", "rb") as fp:
                    imgNames = pickle.load(fp)
                with open("images.txt", "rb") as fp:
                    images = pickle.load(fp)
                # matching the input feature with the loaded images features.
                attendance = []
                for encodeInput, facesOfInput in zip(inputFeatures, faceLocation):
                    # we don't require matches we take distance as first preference for accuracy.
                    # matchs = face_recognition.compare_faces(featuresOfTrainingImages, encodeInput, tolerance=0.3)
                    # print(encodeInput)
                    faceDistance = face_recognition.face_distance(featuresOfTrainingImages, encodeInput)

                    print(imgNames, "names")
                    print(faceDistance, "facedistance")
                    index = np.argmin(faceDistance)
                    print(imgNames[index])
                    if imgNames not in attendance:
                        attendance.append(imgNames[index])
            # cv2.imwrite(os.path.join(finalPath, 'InputImage.jpg'), imgInput)
            #
            # for i in range(len(faceDistance)):
            #     if faceDistance[i] == faceDistance[index]:
            #         print(imgNames[i])
            # for i in range(len(matchs)):
            #     if matchs[i]:
            #         print(i, matchs[i])
            #         cv2.imwrite(os.path.join(finalPath, 'MatchedImage' + str(i) + '.jpg'), images[i])
            #         name = imgNames[i].lower()
            #         flag = 0
            #         print(name)
            # if flag:
            #     flag = 0
            #     c = 1
            #     print("Closely related face")
            #     for i in range(len(faceDistance)):
            #         if faceDistance[i] <= faceDistance[index]:
            #             c = 0
            #             print(imgNames[i])
            #             cv2.imwrite(os.path.join(finalPath, 'CloselyMatchedImage' + str(i) + '.jpg'), images[i])
            #     if c:
            #         print("No Face are Matched With your Input Image")
            #
            # if flag:
            #     print("No Face are Matched With your Input Image")
        except Exception as e:
            print(f"We are sorry! Unable to find the Faces {e}")

cv2.waitKey(0)
