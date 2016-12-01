from neural_net import NeuralNet
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout, Activation, Embedding, merge, Input
from keras.models import Sequential, Model
from keras.utils import np_utils
from data_snli import SNLI
from keras.preprocessing import sequence

class SharedEncoding(NeuralNet):
	def __init__(self, data, num_targets, lstm_size=20, dense_size=32, dense_activation='relu', batch_size = 10000, num_epochs=50):
		super(SharedEncoding,self).__init__(data=data, num_targets=num_targets, lstm_size=lstm_size, dense_size=dense_size, dense_activation=dense_activation, batch_size=batch_size, num_epochs=num_epochs)
		self.maxLength = max(self.maxLengths)

	def sequence_padding(self,data):
		prem, hyp = data
		return [sequence.pad_sequences(prem, dtype='float32',padding='post',maxlen=self.maxLength),sequence.pad_sequences(hyp, dtype='float32',padding='post',maxlen=self.maxLength)]

	def build_model(self):
		premise_input = Input(shape=(self.maxLength, 300), dtype='float32', name='premise')
		hypothesis_input = Input(shape=(self.maxLength, 300), dtype='float32', name='hypothesis')
		
		shared_lstm = LSTM(self.lstm_size)
		encoded_prem= shared_lstm(premise_input)
		encoded_hyp = shared_lstm(hypothesis_input)

		merged_vector = merge([encoded_prem, encoded_hyp], mode='concat', concat_axis=-1)
		x = Dense(self.dense_size,activation=self.dense_activation)(merged_vector)
		x = Dense(self.dense_size,activation=self.dense_activation)(x)
		predictions = Dense(self.num_targets,activation='softmax')(x)
		
		self.model = Model(input=[premise_input, hypothesis_input], output=predictions)
		self.model.compile(optimizer='rmsprop',
		              loss='categorical_crossentropy',
		              metrics=['accuracy'])

class SharedBidirectional(SharedEncoding):
	def build_model(self):
		premise_input = Input(shape=(self.maxLength, 300), dtype='float32', name='premise')
		hypothesis_input = Input(shape=(self.maxLength, 300), dtype='float32', name='hypothesis')
		
		shared_lstm_fwd = LSTM(self.lstm_size)
		shared_lstm_bwd = LSTM(self.lstm_size, go_backwards=True)
		encoded_prem_fwd = shared_lstm_fwd(premise_input)
		encoded_hyp_fwd = shared_lstm_fwd(hypothesis_input)		
		encoded_prem_bwd = shared_lstm_bwd(premise_input)
		encoded_hyp_bwd = shared_lstm_bwd(hypothesis_input)

		merged_vector = merge([encoded_prem_fwd, encoded_prem_bwd, encoded_hyp_fwd, encoded_hyp_bwd], mode='concat')
		x = Dense(self.dense_size,activation=self.dense_activation)(merged_vector)
		x = Dense(self.dense_size,activation=self.dense_activation)(x)
		predictions = Dense(self.num_targets,activation='softmax')(x)
		
		self.model = Model(input=[premise_input, hypothesis_input], output=predictions)
		self.model.compile(optimizer='rmsprop',
		              loss='categorical_crossentropy',
		              metrics=['accuracy'])
