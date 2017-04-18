import numpy as np
import keras
import csv
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *

img_rows = 16
img_cols = 32
data_folder = 'IMG/'
batch_size=128
nb_epoch=30

# resize and extract S channel
def image_preprocessing(img):
	return cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(img_cols,img_rows))

def load_data(X,y,data_folder,delta=0.08):
	"""function to load training data"""

	log_path = 'driving_log.csv'
	logs = []

	# load logs
	with open(log_path,'rt') as f:
		reader = csv.reader(f)
		for line in reader:
			logs.append(line)
		log_labels = logs.pop(0)

	# load center camera image
	for i in range(len(logs)):
		for camera, delta in zip([0,1,2], [0,0.08, -0.08]):
			img_path = data_folder+ logs[i][camera].split('/')[-1]
			img = plt.imread(img_path)
			X.append(image_preprocessing(img))
			y.append(float(logs[i][3]))



if __name__ == '__main__':

	#load data

	print("loading data...")

	data={}
	data['features'] = []
	data['target'] = []

	load_data(data['features'], data['target'],data_folder,0.3)

	X_train = np.array(data['features']).astype('float32')
	y_train = np.array(data['target']).astype('float32')

	# flip images on y axis
	X_train = np.append(X_train,X_train[:,:,::-1],axis=0)
	y_train = np.append(y_train,-y_train,axis=0)

	X_train, y_train = shuffle(X_train, y_train)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.1)

	# reshape to add channel dimension
	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)


	model = Sequential([
			Lambda(lambda x: x/127.5 - 1.,input_shape=(img_rows,img_cols,1)),
			Conv2D(12, 3, input_shape=(img_rows,img_cols,1), activation='relu'),
			MaxPooling2D((2,2)),
			Conv2D(24, 3, input_shape=(img_rows,img_cols,1), activation='relu'),
			MaxPooling2D((2,2)),
			Dropout(0.25),
			Flatten(),
			Dense(100, activation='relu'),
			Dense(50, activation='relu'),
			Dense(1)
		])

	model.summary()

	print("training model...")

	model.compile(loss='mean_squared_error',optimizer='adam')
	history = model.fit(X_train, y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(X_val, y_val))

	print('saving model...')
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights("model.h5")
	print("model saved.")
