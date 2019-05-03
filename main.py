import keras
import numpy as np

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers import Flatten, Lambda, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam as Adam
from keras.layers.advanced_activations import LeakyReLU

import pickle
from collections import defaultdict


from keras.utils import np_utils

from mnist import MNIST

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt


def getData():
	# This loads in the character datasets that we want to interpret
	mndata = MNIST('data')

	x_train, y_train = mndata.load('data/emnist-letters-train-images-idx3-ubyte', 'data/emnist-letters-train-labels-idx1-ubyte')
	x_test, y_test = mndata.load('data/emnist-letters-test-images-idx3-ubyte', 'data/emnist-letters-test-labels-idx1-ubyte')

	# This puts it into the form that we want to pass into keras, (rows, columns, channels)
	# Channels is just 1 because our images are gray-scale
	x_train = np.array(x_train) / 255.0
	y_train = np.array(y_train) - 1
	x_test = np.array(x_test) / 255.0
	y_test = np.array(y_test) - 1

	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)	


	
	y_train = np_utils.to_categorical(y_train, 26)
	y_test = np_utils.to_categorical(y_test, 26)

	# plt.imshow(x_train[0])
	# plt.show()

	return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = getData()




# Set the CNN Architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(26, activation='softmax'))

# Comple the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train,
          epochs=1,
          verbose=1,
          validation_data=(x_test, y_test))
# Save the model weights for future reference
model.save('emnist_cnn_model.h5')

model = load_model('emnist_cnn_model.h5')

# Evaluate the model using Accuracy and Loss
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

