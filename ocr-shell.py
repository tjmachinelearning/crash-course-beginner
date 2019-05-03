from keras.models import load_model

model = load_model('model.h5')


phrases = []


# Given a 28x28x1 np array, it'll add the prediction to the letters that are already predicted
def addLetterToPhrases(img):
	raw_predict = model.predict(img)
	phrases += [chr(raw_predict+65)]


# Code this. Given an input image with a bunch of letters on it,
# it should iterate through and find all letters in order
def generatePhrases(img):




