import cv2
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def classify_image(image_path):

	prototxt = "face_detector/setting.prototxt"
	caffeModel = "face_detector/res10.caffemodel"

	# load input image
	image = cv2.imread(image_path)

	# load our mask detector model
	model = load_model("mask_detector.model")

	# convert image to blob
	blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)), 1.0, (300, 300),
		(104, 177, 123))

	# use caffe model to get detections
	net = cv2.dnn.readNetFromCaffe(prototxt, caffeModel)
	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):
		# extract the confidence and prediction
		confidence = detections[0, 0, i, 2]

		# filter detections by confidence greater than the minimum
		if confidence < 0.5:
			continue

		# compute coordinates of the bounding box for face
		m,n = image.shape[:2]
		box = detections[0, 0, i, 3:7] * np.array([n, m, n, m])
		(x1, y1, x2, y2) = box.astype("int")

		# change color, resize, preprocess
		face = cv2.resize(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), (224, 224))
		face = preprocess_input(img_to_array(face))
		face = face[np.newaxis, :]

		# make prediction
		inproperMask, mask, noMask = model.predict(face)[0]

		#process output image
		pred_value = [inproperMask,mask, noMask]
		classes = ["Inproperly Weared","Mask", "No Mask"]
		colors = [(0,0,255), (0,255,0), (0,0,255)]

		pred_res_idx = np.argmax(pred_value)
		pred_class = classes[pred_res_idx]
		color = colors[pred_res_idx]

		text = "%s: %d%s" % (pred_class, max(pred_value) * 100, '%')

		cv2.putText(image, text, (x1, y1 - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

	# show the output image
	cv2.imshow("output", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.waitKey(1)
	
if __name__ == "__main__":
	image_path = "test.png"
	classify_image(image_path)
