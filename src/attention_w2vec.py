from neural_net import NeuralNet
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout, Activation, Embedding, merge, Input
from keras.models import Sequential, Model
from keras.preprocessing import sequence
from keras.layers.core import Lambda
from collections import Counter
from keras.layers import Dropout
from keras.layers.core import Lambda,RepeatVector,Activation,Flatten,Permute,Reshape
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.np_utils import to_categorical

import data_snli
import w2vec

def get_H_n(X):
    ans = X[:, -1, :]  # get last element from time dim
    return ans

def get_Y(X, xmaxlen):
    return X[:, :xmaxlen, :]  # get first xmaxlen elem from time dim

def get_R(X):
    Y, alpha = X[0], X[1]
    ans = K.T.batched_dot(Y, alpha)
    return ans


class AttentiveLstm(NeuralNet):
	def __init__(self, data, num_targets, lstm_size=20, dense_size=32, dense_activation='relu', batch_size = 10000, num_epochs=50):
		super(AttentiveLstm,self).__init__(data=data, num_targets=num_targets, lstm_size=lstm_size, dense_size=dense_size, dense_activation=dense_activation, batch_size=batch_size, num_epochs=num_epochs)
		#add other variables here and change models
		
	def build_model(self):
		k = 2*self.lstm_size
		L = self.maxLengths[0] +1
		N = self.maxLengths[1] + L
		
		# main_input = Input(shape=(N,), dtype='int32', name='main_input')
		premise_input = Input(shape=(self.maxLengths[0]+1,300), dtype='float32', name='premise')
		hypothesis_input = Input(shape=(self.maxLengths[1],300), dtype='float32', name='hypothesis')
		
		main_input = merge([premise_input, hypothesis_input], mode='concat',concat_axis=1)
		# print main_input.output_shape

		fwd = LSTM(self.lstm_size, return_sequences=True,name='lstm_fwd')

		lstm_fwd = fwd(main_input)
		lstm_bwd = LSTM(self.lstm_size, return_sequences=True, go_backwards=True, name='lstm_bwd')(main_input)
		bilstm = merge([lstm_fwd,lstm_bwd],name='bilstm',mode='concat')
		
		h_n = Lambda(get_H_n, output_shape=(k,),name='h_n')(bilstm)
		Y = Lambda(get_Y, arguments={"xmaxlen":L}, name="Y",output_shape=(L,k))(bilstm)
		Whn = Dense(k, W_regularizer=l2(0.01), name="Wh_n")(h_n)
		Whn_x_e = RepeatVector(L, name="Wh_n_x_e")(Whn)
		WY = TimeDistributed(Dense(k, W_regularizer=l2(0.01)), name="WY")(Y)
		merged = merge([Whn_x_e, WY], name="merged", mode='sum')
		M = Activation('tanh', name="M")(merged)

		alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
		flat_alpha = Flatten(name="flat_alpha")(alpha_)
		alpha = Dense(L, activation='softmax', name="alpha")(flat_alpha)

		Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)
		r_ = merge([Y_trans, alpha], output_shape=(k, 1), name="r_", mode=get_R)
		r = Reshape((k,), name="r")(r_)

		Wr = Dense(k, W_regularizer=l2(0.01))(r)
		Wh = Dense(k, W_regularizer=l2(0.01))(h_n)
		merged = merge([Wr, Wh], mode='sum')
		h_star = Activation('tanh')(merged)
		out = Dense(3, activation='softmax')(h_star)
		output = out

		self.model = Model(input=[premise_input, hypothesis_input], output=output)
		# self.model.summary()
		self.model.compile(loss='categorical_crossentropy',optimizer=Adam(0.001),metrics=['accuracy'])

	def sequence_padding(self,data):
		#delim
		maxlenp = self.maxLengths[0]+1
		maxlenh = self.maxLengths[1]
		prem, hyp = data
		paddedprem, paddedhyp = sequence.pad_sequences(prem, dtype='float32',padding='post',maxlen=maxlenp),sequence.pad_sequences(hyp, dtype='float32',padding='post',maxlen=maxlenh)
		return [paddedprem, paddedhyp]

# w2vec = w2vec.W2Vec()
# snli = data_snli.SNLI(w2vec)

# m = AttentiveLstm(snli,3,10,20,'relu',500,10)
# m.train()
# m.model.save('alstm.h5')
# m.test()