import cv2
import face_recognition
import dlib


capture = cv2.VideoCapture('/dev/video0')

face_locs = []

while True:
	ret, frame = capture.read()
	if ret == False:
		print('*Web-Cam Connection Error*')
		break

	rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	face_locs = face_recognition.face_locations(rgbFrame)

	for top, right, bottom, left in face_locs:
		cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 2)

	cv2.imshow('webcam', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.destroyAllWindows()

