from neural_net import NeuralNet
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout, Activation, Embedding, merge, Input, Reshape
from keras.models import Sequential, Model
from keras.utils import np_utils
from data_snli import SNLI
from keras.preprocessing import sequence

class ConditionalEncoding(NeuralNet):

	def __init__(self, data, num_targets):
		super(ConditionalEncoding,self).__init__(data)
		#add other variables here and change models
		self.num_targets = num_targets

	def sequence_padding(self):
		self.maxLength = max(self.data.getMaxLengths())
		self.premise_train = sequence.pad_sequences(self.premise_train, dtype='float32',padding='post',maxlen=self.maxLength)
		self.hypothesis_train = sequence.pad_sequences(self.hypothesis_train,dtype='float32',padding='post',maxlen=self.maxLength)
		self.premise_test= sequence.pad_sequences(self.premise_test, dtype='float32',padding='post',maxlen=self.maxLength)
		self.hypothesis_test = sequence.pad_sequences(self.hypothesis_test,dtype='float32',padding='post',maxlen=self.maxLength)

	def build_model(self):
		premise_input = Input(shape=(self.premise_train[0].shape), dtype='float32', name='premise')
		hypothesis_input = Input(shape=(self.hypothesis_train[0].shape), dtype='float32', name='hypothesis')

		premise_encoded = LSTM(50)(premise_input)

		num_dim_shared_lstm = 50
		shared_lstm = LSTM(num_dim_shared_lstm)
		reshaped_premise = Reshape((1,num_dim_shared_lstm))(premise_encoded)
		encoded_prem_lstm2 = shared_lstm(reshaped_premise)
		both_encoded = shared_lstm(hypothesis_input)

		x = Dense(64,activation='relu')(both_encoded)
		x = Dense(64,activation='relu')(x)
		predictions = Dense(self.num_targets,activation='softmax')(x)

		self.model = Model(input=[self.premise_input, self.hyp_input], output=predictions)
		self.model.compile(optimizer='rmsprop',
		              loss='categorical_crossentropy',
		              metrics=['accuracy'])

# c = ConditionalEncoding(SNLI(),3)