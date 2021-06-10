from imutils import face_utils
import imutils
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
while True:
    re, frame = cap.read()
    if re == False:
        print('*Web-Cam Connection Error*')
        break
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        faceHull = cv2.convexHull(shape)
        cv2.drawContours(frame, [faceHull], -1, (0, 0, 255), 1)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow('Primat', frame)
    key = cv2.waitKey(1) & 0xFF
    if key==27 or re==False:
        break
cap.release()
cv2.destroyAllWindows()