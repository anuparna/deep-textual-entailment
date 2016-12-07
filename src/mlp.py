from neural_net import NeuralNet
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout, Activation, Embedding, merge, Input
from keras.models import Sequential, Model
from keras.preprocessing import sequence
import sys
import data_snli
import w2vec


class Mlp(NeuralNet):
	def __init__(self, data, num_targets, lstm_size=20, dense_size=32, dense_activation='relu', batch_size = 100, num_epochs=10):
		super(Mlp,self).__init__(data=data, num_targets=num_targets, lstm_size=lstm_size, dense_size=dense_size, dense_activation=dense_activation, batch_size=batch_size, num_epochs=num_epochs)
		#add other variables here and change models
		
	def build_model(self):
		premise_input = Input(shape=(self.maxLengths[0],300), dtype='float32', name='premise')
		hyp_input = Input(shape=(self.maxLengths[1],300), dtype='float32', name='hypothesis')
		
		inputs_encoded = merge([premise_input, hyp_input], mode='concat')

		x = Dense(self.dense_size,activation=self.dense_activation)(inputs_encoded)
		x = Dense(self.dense_size,activation=self.dense_activation)(x)
		predictions = Dense(self.num_targets, activation='softmax')(x)

		self.model = Model(input=[premise_input, hyp_input], output=predictions)
		self.model.compile(optimizer='rmsprop',
		              loss='categorical_crossentropy',
		       	       metrics=['accuracy'])
		

	def sequence_padding(self,data):
		# maxlenp = self.maxLengths[0]
		# maxlenh = self.maxLengths[1]
		# prem, hyp = data
		# paddedprem, paddedhyp = sequence.pad_sequences(prem, dtype='float32',padding='post',maxlen=maxlenp),sequence.pad_sequences(hyp, dtype='float32',padding='post',maxlen=maxlenh)
		return data



w2vec = w2vec.W2Vec()
snli = data_snli.SNLI(w2vec)

m = Mlp(snli,3,20,200,'relu',5000,50)
m.train()
m.model.save('mlp.h5')
m.test()
