import cv2
import face_recognition
import os
import numpy as np

# folderName = input("Enter a folder name for your Search : ")
# try:
#     os.mkdir(folderName)
# except:
#     os.rmdir(folderName)
#     os.mkdir(folderName)
# imgPath = input("Give the Path of the Image : ")
#
# imgInput = face_recognition.load_image_file(imgPath)
# imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)

path = "E:\PycharmProjects\/attendence\MainImages"
images = []
imgNames = []
# Now we have to get the files from the MainImages Folder
myPhotos = os.listdir(path)
print(myPhotos)

for img in myPhotos:
    imgs = cv2.imread(f'{path}/{img}')
    images.append(imgs)
    imgNames.append(os.path.splitext(img)[0])
print(imgNames, "Imagnmaes")


def features(images):
    featuresOfImages = []
    c = 0
    for img in images:
        c += 1
        imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(face_recognition.face_locations(imgs)[0])
        try:
            featuresOfImg = face_recognition.face_encodings(imgs, model='cnn')[0]
        except IndexError as e:
            print("Some Faces are not detected by dlib")
            # sys.exit(1)
        featuresOfImages.append(featuresOfImg)
        print(c, "c")
    return featuresOfImages


featuresOfTrainingImages = features(images)
print(type(featuresOfTrainingImages))
print("Features are collected...", len(featuresOfTrainingImages))

imgPath = input("Give the Path of the Image : ")
imgInput = face_recognition.load_image_file(imgPath)
imgInput = cv2.resize(imgInput, (0, 0), None, 0.25, 0.25)
imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)
faceLocation = face_recognition.face_locations(imgInput)
print(faceLocation, "input face")
inputFeatures = face_recognition.face_encodings(imgInput, model='cnn')[0]
print(len(inputFeatures))

# matchs = face_recognition.compare_faces([featuresOfTrainingImages], inputFeatures)
# print(matchs)
faceDistance = face_recognition.face_distance(featuresOfTrainingImages, inputFeatures)
print(faceDistance,"facedistance")
index = np.argmin(faceDistance)
print(index,"index")
name = imgNames[index].lower()
print(name)
