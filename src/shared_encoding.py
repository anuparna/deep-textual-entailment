from neural_net import NeuralNet
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout, Activation, Embedding, merge, Input
from keras.models import Sequential, Model
from keras.utils import np_utils
from data_snli import SNLI
from keras.preprocessing import sequence

class SharedEncoding(NeuralNet):
	def __init__(self, data, num_targets):
		super(SharedEncoding,self).__init__(data)
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
		
		shared_lstm = LSTM(50)
		
		encoded_prem= shared_lstm(premise_input)
		encoded_hyp = shared_lstm(hypothesis_input)

		merged_vector = merge([encoded_prem, encoded_hyp], mode='concat', concat_axis=-1)
		x = Dense(64,activation='relu')(merged_vector)
		x = Dense(64,activation='relu')(x)
		predictions = Dense(self.num_targets,activation='softmax')(x)

		self.model = Model(input=[premise_input, hypothesis_input], output=predictions)
		self.model.compile(optimizer='rmsprop',
		              loss='categorical_crossentropy',
		              metrics=['accuracy'])

# c = SharedEncoding(SNLI(),3)