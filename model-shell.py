import keras
import numpy as np
from keras.utils import np_utils
from mnist import MNIST
from matplotlib import pyplot as plt

def getData():
	# This loads in the character datasets that we want to interpret
	mndata = MNIST('data')
	x_train, y_train = mndata.load('data/emnist-letters-train-images-idx3-ubyte', 'data/emnist-letters-train-labels-idx1-ubyte')
	x_test, y_test = mndata.load('data/emnist-letters-test-images-idx3-ubyte', 'data/emnist-letters-test-labels-idx1-ubyte')

	print("Finished loading data.\n")
	# Normalizes our data from 0 to 255 to 0 to 1
	x_train = np.array(x_train) / 255.0
	x_test = np.array(x_test) / 255.0

	# The data given is from 1 to 26, so this 0 indexes it
	y_train = np.array(y_train) - 1
	y_test = np.array(y_test) - 1

	# This puts it into the form that we want to pass into keras (samples, rows, columns, channels)
	# Channels is just 1 because our images are gray-scale
	# x_train.shape[0] tells how many samples we have
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)	


	# Rotates the image and reflects it so it looks "normal" to us
	x_train = np.rot90(x_train, axes=(1,2), k=3)
	x_test = np.rot90(x_test, axes=(1,2), k=3)
	x_train = x_train[:,:,::-1,:]
	x_test = x_test[:,:,::-1,:]


	# This just shows how some of our samples look like, comment it out after trying it out
	for z in range(0,10):
		fig = plt.figure()
		plt.imshow(255*x_train[z].reshape(28,28), interpolation="nearest", cmap="gray")
		fig.suptitle(chr(y_train[z]+65), fontsize=20)
		plt.show()


	# This one-hot encodes our data. This post explains why we need to do it, if you're interested
	# https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f
	y_train = np_utils.to_categorical(y_train, 26)
	y_test = np_utils.to_categorical(y_test, 26)


	return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = getData()


# Build, compile, and fit your model. At the end, save your model to an h5 file





