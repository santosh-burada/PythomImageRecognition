import cv2
import face_recognition

imgSantosh = face_recognition.load_image_file('Images/santosh.jpg')
imgSantosh = cv2.cvtColor(imgSantosh, cv2.COLOR_BGR2RGB)

imgSantoshTest = face_recognition.load_image_file('Images/SANTOSH-TEST.jpg')
imgSantoshTest = cv2.cvtColor(imgSantoshTest, cv2.COLOR_BGR2RGB)

faceLocation = face_recognition.face_locations(imgSantosh)[0]
encodeSantosh = face_recognition.face_encodings(imgSantosh)[0]
cv2.rectangle(imgSantosh, (faceLocation[3], faceLocation[0]),
              (faceLocation[1], faceLocation[2]), (0, 0, 255), 2)

faceLocationTest = face_recognition.face_locations(imgSantoshTest)[0]
encodeSantoshTest = face_recognition.face_encodings(imgSantoshTest)[0]
cv2.rectangle(imgSantoshTest, (faceLocationTest[3], faceLocationTest[0]),
              (faceLocationTest[1], faceLocationTest[2]), (0, 0, 255), 2)

results = face_recognition.compare_faces([encodeSantosh],encodeSantoshTest)
if results[0]:
    print("It is Santosh")
else:
    print("Unknow")

print(faceLocation,"faceMain")
print(faceLocationTest,"test")

imgSantosh = cv2.resize(imgSantosh,(500,700))
imgSantoshTest = cv2.resize(imgSantoshTest,(500,700))
cv2.imshow("Main", imgSantosh)
cv2.imshow("Test", imgSantoshTest)
cv2.waitKey(0)
