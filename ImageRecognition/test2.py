# import cv2
#
# img = cv2.imread('E:\PycharmProjects\attendence\InputImages\Dr.LakshmanRao.jpg')
# cv2.imshow("image", img)
# cv2.waitKey(0)

# !/usr/bin/python3
# import os
# import cv2
#
# os.chdir("E:\PycharmProjects\/attendence\TestImages")
# for root, dirs, files in os.walk(".", topdown=False):
#
#     for name in files:
#         imgNames.append(name.split(".")[0])
#         images.append(cv2.imread(os.path.join(root, name)))

import cv2
import face_recognition
import os
import numpy as np
import shutil
import pickle

print("For Search-0")
print("For load-1")

n = int(input("[0/1]"))

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
    print("NOTE :- This Folder is Used to Store your Input Image and matched Images")
    folderName = input("Enter a folder name for your Search : ")
    pathOfFolder = input(f"Enter the Storage Path to store the {folderName} : ")

    finalPath = os.path.join(pathOfFolder, folderName)
    try:
        os.mkdir(finalPath)
    except:
        shutil.rmtree(finalPath)
        os.mkdir(finalPath)

    imgPath = input("Give the Path of the Input Image : ")
    imgInput = face_recognition.load_image_file(imgPath)
    imgInput = cv2.resize(imgInput, (0, 0), None, 0.25, 0.25)
    imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)
    # cv2.imshow("input image", imgInput)
    try:

        faceLocation = face_recognition.face_locations(imgInput)
        print(faceLocation, "input face location")
        inputFeatures = face_recognition.face_encodings(imgInput, faceLocation, model='cnn')
        print("saomething")
        # Here we wil have the input face features
        with open("featuresOfTrainingImages.txt", "rb") as fp:  # Unpickling
            featuresOfTrainingImages = pickle.load(fp)
        with open("imgNames.txt", "rb") as fp:  # Unpickling
            imgNames = pickle.load(fp)
        with open("images.txt", "rb") as fp:  # Unpickling
            images = pickle.load(fp)
        # matching the input feature with the loaded images features.
        print("saomething")
        attendance = []
        for encodeInput, facesOfInput in zip(inputFeatures, faceLocation):
            print("saomething1")
            # we don't require matches we take distance as first preference for accuracy.
            # matchs = face_recognition.compare_faces(featuresOfTrainingImages, encodeInput, tolerance=0.3)
            # print(encodeInput)
            faceDistance = face_recognition.face_distance(featuresOfTrainingImages, encodeInput)
            # print(matchs, "These are matching with Loaded images from MainImages folder")

            print(imgNames, "names")
            print(faceDistance, "facedistance")
            index = np.argmin(faceDistance)
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
