import numpy as np

from keras.preprocessing import sequence

from data_snli import SNLI
from abc import ABCMeta, abstractmethod

class NeuralNet:
	__metaclass__ = ABCMeta

	def __init__(self, data):
		(self.premise_train, self.hypothesis_train), self.y_train = data.getData('train')
		(self.premise_test, self.hypothesis_test), self.y_test = data.getData('test')
		self.maxLengths = data.getMaxLengths()

		self.premise_train = sequence.pad_sequences(self.premise_train, dtype='float32',padding='post',maxlen=maxLengths[0])
		self.hypothesis_train = sequence.pad_sequences(self.hypothesis_train,dtype='float32',padding='post',maxlen=maxLengths[1])

		self.premise_test= sequence.pad_sequences(self.premise_test, dtype='float32',padding='post',maxLen=maxLengths[0])
		self.hypothesis_test = sequence.pad_sequences(self.hypothesis_test,dtype='float32',padding='post',maxLen=maxLengths[1])

	def train(self):
		self.model.fit([self.premise_train, self.hypothesis_train], self.y_train)

	def test(self):
		print self.model.evaluate([self.premise_test, self.hypothesis_test],self.y_test)

	@abstractmethod
	def buildModel(self):
		pass

