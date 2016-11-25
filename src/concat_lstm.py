from neural_net import NeuralNet
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout, Activation, Embedding, merge, Input
from keras.models import Sequential, Model
from keras.utils import np_utils
from data_snli import SNLI
from w2vec import W2Vec
from keras.preprocessing import sequence

import random
class ConcatLstm:
	def __init__(self, data, num_targets):
		#add other variables here and change models
		self.num_targets = num_targets
		# (self.premise_train, self.hypothesis_train), self.y_train = data.getData('train')
		# (self.premise_test, self.hypothesis_test), self.y_test = data.getData('test')
		self.data = data
		self.maxLengths = data.getMaxLengths()

	def build_model(self):
		premise_input = Input(shape=(self.maxLengths[0],300), dtype='float32', name='premise')
		premise_encoded = LSTM(20)(premise_input)

		hyp_input = Input(shape=(self.maxLengths[1],300), dtype='float32', name='hypothesis')
		hyp_encoded = LSTM(20)(hyp_input)
		
		inputs_encoded = merge([premise_encoded, hyp_encoded], mode='concat')

		x = Dense(32,activation='relu')(inputs_encoded)
		x = Dense(32,activation='relu')(x)
		predictions = Dense(self.num_targets,activation='softmax')(x)

		self.model = Model(input=[premise_input, hyp_input], output=predictions)
		self.model.compile(optimizer='rmsprop',
		              loss='categorical_crossentropy',
		              metrics=['accuracy'])

	# def sequence_padding(self):

	# 	self.premise_train = sequence.pad_sequences(self.premise_train, dtype='float32',padding='post',maxlen=self.maxLengths[0])
	# 	self.hypothesis_train = sequence.pad_sequences(self.hypothesis_train,dtype='float32',padding='post',maxlen=self.maxLengths[1])

	# 	self.premise_test= sequence.pad_sequences(self.premise_test, dtype='float32',padding='post',maxlen=self.maxLengths[0])
	# 	self.hypothesis_test = sequence.pad_sequences(self.hypothesis_test,dtype='float32',padding='post',maxlen=self.maxLengths[1])

	def train(self):
		self.model.fit_generator(self.batch_generator('train',100), 500000, 10)

	def test(self):
		print self.model.evaluate_generator(self.batch_generator('test',100), 10000)

	def batch_generator(self, dataset, batch_size):
		while True:
			y = self.data.getY('train')
			for i in range(len(y)):
				start_index = (i*batch_size)
				end_index = ((i+1)*batch_size)
				# print start_index, end_index
				if end_index>len(y):
					break
				yield (self.sequence_padding(self.data.getX(dataset, start_index, end_index)), self.data.getY(dataset,start_index,end_index))

	def sequence_padding(self,data):
		maxlenp = self.maxLengths[0]
		maxlenh = self.maxLengths[1]
		prem,hyp = data
		# print prem.shape, hyp.shape
		paddedprem, paddedhyp = sequence.pad_sequences(prem, dtype='float32',padding='post',maxlen=maxlenp),sequence.pad_sequences(hyp, dtype='float32',padding='post',maxlen=maxlenh)
		# print paddedprem.shape, paddedhyp.shape
		# print len(paddedprem), len(paddedhyp)
		return [paddedprem, paddedhyp]

# c = ConcatLstm(SNLI(W2Vec()),3)
# c.build_model()
# c.train()
# c.test()