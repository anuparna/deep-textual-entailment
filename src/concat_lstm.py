from neural_net import NeuralNet
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout, Activation, Embedding, merge, Input
from keras.models import Sequential, Model
from keras.utils import np_utils
from data_snli import SNLI
from w2vec import W2Vec
from keras.preprocessing import sequence

class ConcatLstm:
	def __init__(self, data, num_targets, lstm_size=20, dense_size=32, dense_activation='relu', batch_size = 100, num_epochs=10):
		#add other variables here and change models
		self.num_targets = num_targets
		self.data = data
		self.maxLengths = data.getMaxLengths()

		self.lstm_size = lstm_size
		self.dense_size = dense_size
		self.dense_activation = dense_activation
		self.batch_size = batch_size
		self.num_epochs = num_epochs

		# since we are ignoring samples at the end, if not enough to form a full batch. see logic in batch_generator
		self.train_size = (len(self.data.data['X']['train'])/(self.batch_size))*self.batch_size
		self.test_size = (len(self.data.data['X']['test'])/(self.batch_size))*self.batch_size
	

	def build_model(self):
		premise_input = Input(shape=(self.maxLengths[0],300), dtype='float32', name='premise')
		premise_encoded = LSTM(self.lstm_size)(premise_input)

		hyp_input = Input(shape=(self.maxLengths[1],300), dtype='float32', name='hypothesis')
		hyp_encoded = LSTM(self.lstm_size)(hyp_input)
		
		inputs_encoded = merge([premise_encoded, hyp_encoded], mode='concat')

		x = Dense(self.dense_size,activation=self.dense_activation)(inputs_encoded)
		x = Dense(self.dense_size,activation=self.dense_activation)(x)
		predictions = Dense(self.num_targets, activation='softmax')(x)

		self.model = Model(input=[premise_input, hyp_input], output=predictions)
		self.model.compile(optimizer='rmsprop',
		              loss='categorical_crossentropy',
		              metrics=['accuracy'])

	def train(self):
		self.build_model()
		self.model.fit_generator(self.batch_generator('train'), self.train_size, self.num_epochs)

	def test(self):
		print self.model.evaluate_generator(self.batch_generator('test'), self.test_size)

	def batch_generator(self, dataset):
		while True:
			y = self.data.getY(dataset)
			for i in range(len(y)):
				start_index = (i*self.batch_size)
				end_index = ((i+1)*self.batch_size)
				if end_index>=len(y):
					break
				yield (self.sequence_padding(self.data.getX(dataset, start_index, end_index)), self.data.getY(dataset,start_index,end_index))

	def sequence_padding(self,data):
		maxlenp = self.maxLengths[0]
		maxlenh = self.maxLengths[1]
		prem, hyp = data
		paddedprem, paddedhyp = sequence.pad_sequences(prem, dtype='float32',padding='post',maxlen=maxlenp),sequence.pad_sequences(hyp, dtype='float32',padding='post',maxlen=maxlenh)
		return [paddedprem, paddedhyp]
