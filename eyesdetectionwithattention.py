from imutils import face_utils
import imutils
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('resources/shape_predictor_68_face_landmarks.dat')



cap = cv2.VideoCapture(0)
while True:
    re, frame = cap.read()
    if re == False:
        print('*Web-Cam Connection Error*')
        break
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        ll = shape[28][0]-shape[18][0]
        lw = ll
        left_eye = frame[shape[19][1]:shape[19][1]+lw, shape[17][0]:shape[17][0]+ll]
        right_eye = frame[shape[25][1]:shape[25][1] + lw, shape[27][0]:shape[27][0] + ll]
        left_eye = imutils.resize(left_eye, width=100)
        right_eye = imutils.resize(right_eye, width=100)
        rows, cols, channels = left_eye.shape
        try:
            #frame[30:130,30:130] = right_eye
            frame[shape[19][1]:shape[19][1]+100, shape[17][0]:shape[17][0]+100] = right_eye
            frame[shape[25][1]:shape[25][1] + 100, shape[27][0]:shape[27][0] + 100] = left_eye
            #frame[30:130,130:230] = left_eye
        except:
            print("Recognition error")

        #print(shape[0][0])

        #for (x, y) in shape:
            # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.circle(frame, (shape[39][0],shape[39][1]), 1, (0, 0, 255), -1)
    cv2.imshow('Primat', frame)
    key = cv2.waitKey(1) & 0xFF
    if key==27 or re==False:
        break
cap.release()
cv2.destroyAllWindows()