from imutils import face_utils
import imutils
import datetime
import argparse
import numpy
import dlib
import cv2
import time
import random

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('resources//shape_predictor_68_face_landmarks.dat')
img = cv2.imread('bully.jpg')
err=0

cap = cv2.VideoCapture(0)

hstartT = 0
vstartT = 0

while True:
    re, frame = cap.read()
    if re is False:
        print('Web-Cam Connection Error')
        break
    frame = imutils.resize(frame, height=600, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        rad1 = abs(shape[33][1]-shape[26][1])
        rad2 = abs(shape[33][0]-shape[26][0])
        rad = int((rad1*rad1+rad2*rad2)**0.55)

        mask = imutils.resize(img, width = int(2.5*rad))
        rows, cols, channels = mask.shape

        # try:
        #     hstart = shape[0][0]
        #     vstart = shape[19][1]
        #     hstart -= rad // 2
        #     vstart -= rad
        #
        #     vend = vstart + rows
        #     hend = hstart+cols
        #     if hend > 800:
        #         hend = 800
        #     elif hend < 0:
        #         hend = 0
        #     if vend > 600:
        #         vend = 600
        #     elif vend < 0:
        #         vend = 0
        #
        #     frame[vstart:vend, hstart:hend] = mask
        # except ValueError:
        #     print(f"Face-Encryption Error  №{err}")
        #     err+=1
        faceCenter = shape[33]
        cv2.circle(frame, (faceCenter[0],faceCenter[1]), rad, (50,50,255), -1)
    #cv2.putText(frame, "Hello_world!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or re == False:
        break
cap.release()
cv2.destroyAllWindows()