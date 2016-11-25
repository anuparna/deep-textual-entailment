from neural_net import NeuralNet
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout, Activation, Embedding, merge, Input
from keras.models import Sequential, Model
from keras.utils import np_utils
from data_snli import SNLI

class ConcatLstm(NeuralNet):
	def __init__(self, data, num_targets):
		super(ConcatLstm,self).__init__(data)
		#add other variables here and change models
		self.num_targets = num_targets

	def build_model(self):
		premise_input = Input(shape=(self.premise_train[0].shape), dtype='float32', name='premise')
		premise_encoded = LSTM(20)(premise_input)

		hyp_input = Input(shape=(self.hypothesis_train[0].shape), dtype='float32', name='hypothesis')
		hyp_encoded = LSTM(20)(hyp_input)
		
		inputs_encoded = merge([premise_encoded, hyp_encoded], mode='concat')

		x = Dense(32,activation='relu')(inputs_encoded)
		x = Dense(32,activation='relu')(x)
		predictions = Dense(self.num_targets,activation='softmax')(x)

		self.model = Model(input=[premise_input, hyp_input], output=predictions)
		self.model.compile(optimizer='rmsprop',
		              loss='categorical_crossentropy',
		              metrics=['accuracy'])


