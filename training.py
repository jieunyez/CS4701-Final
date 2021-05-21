from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import os

imagePaths = list(paths.list_images("dataset"))
data = []
labels = []
for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]

	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	data.append(image)
	labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

labelEncoder = LabelEncoder()
labels = labelEncoder.fit_transform(labels)
labels = to_categorical(labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=30)


resnet50 = ResNet50V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

mask_model = resnet50.output
mask_model = MaxPooling2D(pool_size=(7, 7))(mask_model)
mask_model = Flatten(name="flatten")(mask_model)
mask_model = Dense(128, activation="relu")(mask_model)
mask_model = Dropout(0.5)(mask_model)
mask_model = Dense(64, activation="relu")(mask_model)
mask_model = Dropout(0.3)(mask_model)
mask_model = Dense(32, activation="relu")(mask_model)
mask_model = Dropout(0.3)(mask_model)
mask_model = Dense(3, activation="softmax")(mask_model)


mask_model = Model(inputs=resnet50.input, outputs=mask_model)

for layer in resnet50.layers:
	layer.trainable = False

learningRate = 1e-4
epochs = 10
batchSize = 32

optimizer = Adam(lr=learningRate, decay=learningRate / epochs)
mask_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

augmentation = ImageDataGenerator(
	width_shift_range=0.2,
	height_shift_range=0.2,
	rotation_range=20,
	zoom_range=0.15,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

mask_model.fit(
	augmentation.flow(trainX, trainY, batch_size=batchSize),
	steps_per_epoch=len(trainX) / batchSize,
	validation_data=(testX, testY),
	validation_steps=len(testX) / batchSize,
	epochs=epochs)


pred = mask_model.predict(testX, batch_size=batchSize)
pred = np.argmax(pred, axis=1)

print(classification_report(testY.argmax(axis=1), pred,
	target_names=labelEncoder.classes_))

mask_model.save("mask_detector.model", save_format="h5")
