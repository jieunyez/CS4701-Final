
import numpy as np
import imutils
import time
import cv2
from imutils.video import VideoStream

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def detect_and_predict_mask(win,face_net, mask_net):
    
	ht, wt = win.shape[:2]
	blob = cv2.dnn.blobFromImage(win, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	face_net.setInput(blob)
	detected = face_net.forward()

	allf, locs, preds = [], [], []

	for i in range(0, detected.shape[2]):
		confi = detected[0, 0, i, 2]

		min_confi = 0.5
		if confi > min_confi:
			boundary = detected[0, 0, i, 3:7] * np.array([wt, ht, wt, ht])
			s_x, s_y, e_x, e_y = boundary.astype("int")

			s_x, s_y = (max(0, s_x), max(0, s_y))
			e_x, e_y = (min(wt - 1, e_x), min(ht - 1, e_y))

			face = win[s_y:e_y, s_x:e_x]
			face = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (224,224))
			face = preprocess_input(img_to_array(face))
		
			allf.append(face)
			locs.append((s_x, s_y, e_x, e_y))

	if len(allf) == 1:
		preds = mask_net.predict(np.array(allf, dtype="float32") , batch_size=32)
  
	return (locs, preds)

proto = "setting.prototxt"
weights = "res10.caffemodel"

face_net = cv2.dnn.readNet(proto, weights)

mask_net = load_model("mask_detector.model")

vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	win = vs.read()
	win = imutils.resize(win, width=800)

	(locs, preds) = detect_and_predict_mask(win,
	face_net, mask_net)

	for (boundary, pred) in zip(locs, preds):
		(s_x, s_y, e_x, e_y) = boundary
		(inappropriateMaskWearing, mask, withoutMask) = pred

		label = "Inapproriate Mask Wearing"
		if mask > withoutMask and mask > inappropriateMaskWearing:
			label = "Mask"
		elif withoutMask > mask and withoutMask > inappropriateMaskWearing:
			label = "No Mask"

		color = (255,0,0)
		if label == "No Mask":
			color = (0, 0, 255)
		elif label == "Mask":
			color = (0,255,0)
			
		label = label + ": " + str(round(max(mask, withoutMask, inappropriateMaskWearing) * 100,2))

		cv2.putText(win, label, (s_x, s_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(win, (s_x, s_y), (e_x, e_y), color, 2)

	# show the output win
	cv2.imshow("win", win)
	key = cv2.waitKey(1) & 0xFF

