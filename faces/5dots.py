
import cv2
import dlib 
import imutils
from imutils import face_utils

predictor = dlib.shape_predictor("../resources/shape_predictor_5_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

video_capture = cv2.VideoCapture(0)


while True:
	ret, frame = video_capture.read()

	if ret == False:
		print("Ошибка доступа к камере")
		break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		hull = cv2.convexHull(shape)
		cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

	cv2.imshow("5dots", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()

cv2.destroyAllWindows()
