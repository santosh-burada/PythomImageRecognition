import cv2
import face_recognition
import os
import numpy as np
import shutil

print("NOTE :- This Folder is Used to Store your Input Images and matched Images")
folderName = input("Enter a folder name for your Search : ")
pathOfFolder = input(f"Enter the Storage Path to store the {folderName} : ")

finalPath = os.path.join(pathOfFolder, folderName)
try:
    os.mkdir(finalPath)
except:
    shutil.rmtree(finalPath)
    os.mkdir(finalPath)
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

# Here we are getting the images and appending them to images list and there names to imageNames list
for img in myPhotos:
    imgs = cv2.imread(f'{path}/{img}')
    images.append(imgs)
    imgNames.append(os.path.splitext(img)[0])
print(imgNames, "Imagnmaes")


# cv2.imshow("folder images", images[1])


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


featuresOfTrainingImages = features(images)  # In this featuresOfTrainingImages list we will have the features of all
# loaded images
print(type(featuresOfTrainingImages))
print("Features are collected...", len(featuresOfTrainingImages))

# From here we are taking Image input from the user. Here the user has to give the image path
imgPath = input("Give the Path of the Input Image : ")
imgInput = face_recognition.load_image_file(imgPath)
imgInput = cv2.resize(imgInput, (0, 0), None, 0.25, 0.25)
imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)
# cv2.imshow("input image", imgInput)
faceLocation = face_recognition.face_locations(imgInput)[0]
print(faceLocation, "input face location")
inputFeatures = face_recognition.face_encodings(imgInput, model='cnn')[0]  # Here we wil have the input face features

# matching the input feature with the loaded images features.
matchs = face_recognition.compare_faces(featuresOfTrainingImages, inputFeatures, tolerance=0.5)
print(matchs, "These are matching with Loaded images from MainImages folder")
faceDistance = face_recognition.face_distance(featuresOfTrainingImages, inputFeatures)
print(faceDistance, "facedistance")
index = np.argmin(faceDistance)
cv2.imwrite(os.path.join(finalPath, 'InputImage.jpg'), imgInput)
flag = 1
for i in range(len(matchs)):
    if matchs[i]:
        print(i,matchs[i])
        cv2.imwrite(os.path.join(finalPath, 'MatchedImage' + str(i) + '.jpg'), images[i])
        name = imgNames[i].lower()
        flag = 0
        print(name)
if flag:
    print("No Face are Matched With your Input Image")

# if matchs[index]:
#     name = imgNames[index].lower()
#     print(name)
#     cv2.imwrite(os.path.join(finalPath, 'InputImage.jpg'), imgInput)
#     cv2.imwrite(os.path.join(finalPath, 'MatchedImage.jpg'), images[index])
# else:
#     print("No Face are Matched With your Input")
cv2.waitKey(0)

