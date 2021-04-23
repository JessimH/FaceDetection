import cv2

#choose an image to detect face in
#img = cv2.imread('chat.jpg')

#to capture from camera
webcam = cv2.VideoCapture(0)

while True:
    (read_successful, frame) = webcam.read()

    if read_successful:
        # change la couleur du frame en noir et blanc
        black_n_white_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # pre-trained data on frontal faces from opencv github (haar cascad algorithm)
    trained_face_data = cv2.CascadeClassifier('facedetector.xml')

    # détection de visages
    faces = trained_face_data.detectMultiScale(black_n_white_image)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Face detector', frame)
    key = cv2.waitKey(1)  # 1milisecond

    if key == 81 or key == 113:
        break