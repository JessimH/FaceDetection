import cv2

#choose an image to detect face in
img = cv2.imread('chat.jpg')

#convertion de l'image en noir et blanc (nécessaire pour le "haar Cascade")
black_n_white_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#pre-trained data on frontal faces from opencv github (haar cascad algorithm)
trained_face_data = cv2.CascadeClassifier('facedetector.xml')

#détection de voitures
faces = trained_face_data.detectMultiScale(black_n_white_image)

print(faces)

#Dessin des rectangles sur les voitures détécté
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)

#show image
cv2.imshow('Face detector', img)
cv2.waitKey()

print('code Complete')