
import cv2
from predict import detect
import pickle

filename = 'finalized_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


cap = cv2.VideoCapture(0)


for i in range(100000):

	ret, img = cap.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	cropped_img = faces
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 0), 4)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		cropped_img = img[y:y + h, x:x + w]

	font = cv2.FONT_HERSHEY_SIMPLEX

	org = (50, 50)

	fontScale = 1

	color = (255, 0, 0)

	thickness = 2
	try:
		is_smile = detect(cropped_img, loaded_model)[0]
		if is_smile[0] < 0.7:
			img = cv2.putText(img, 'smile', org, font, fontScale, color, thickness, cv2.LINE_AA)
		else:
			img = cv2.putText(img, 'Non smile', org, font, fontScale, color, thickness, cv2.LINE_AA)
	except:
		pass

	cv2.imshow('live', img)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break


cap.release()

cv2.destroyAllWindows()
