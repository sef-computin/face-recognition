import cv2
import face_recognition
import dlib
import imutils
from imutils import face_utils


predictor = dlib.shape_predictor("../resources/shape_predictor_68_face_landmarks.dat")
capture = cv2.VideoCapture('/dev/video0')
detector = dlib.get_frontal_face_detector()

face_locs = []

while True:
	ret, frame = capture.read()
	if ret == False:
		print('*Web-Cam Connection Error*')
		break

	grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(grayFrame, 0)
	for rect in rects:
		shape = predictor(grayFrame, rect)
		shape = face_utils.shape_to_np(shape)
		hull = cv2.convexHull(shape)

		top = shape[19][1] 
		left = shape[0][0]
		right = shape[15][0]
		bottom = shape[8][1]


		cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 2) 


	cv2.imshow('webcam', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.destroyAllWindows()

