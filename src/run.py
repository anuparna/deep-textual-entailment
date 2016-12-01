import data_snli
import w2vec

w2vec = w2vec.W2Vec()
snli = data_snli.SNLI(w2vec)

import sys

import time
current_milli_time = lambda: int(round(time.time() * 1000))

if len(sys.argv) == 8:
	print 'lstmsize:',sys.argv[1]
	print 'densesize:',sys.argv[2]
	print 'denseactivation:', sys.argv[3]
	print 'num_epochs:',sys.argv[4]
	print 'batch size:',sys.argv[5]
	print 'load:',sys.argv[6]
	print 'model:',sys.argv[7]

	if sys.argv[7]=='ConcatLstm':
		import concat_lstm
		c = concat_lstm.ConcatLstm(snli, 3, int(sys.argv[1]), int(sys.argv[2]), dense_activation=sys.argv[3], batch_size=10000, num_epochs=int(sys.argv[4]))
	elif sys.argv[7]=='SharedEncoding':
		import shared_encoding
		c = shared_encoding.SharedEncoding(snli, 3, int(sys.argv[1]), int(sys.argv[2]), dense_activation=sys.argv[3], batch_size=10000, num_epochs=int(sys.argv[4]))
	elif sys.argv[7]=='SharedBidirectional':
		import shared_encoding
		c = shared_encoding.SharedBidirectional(snli, 3, int(sys.argv[1]), int(sys.argv[2]), dense_activation=sys.argv[3], batch_size=10000, num_epochs=int(sys.argv[4]))
	elif sys.argv[7] == 'Attention':
		import attention_w2vec
		c = attention_w2vec.AttentiveLstm(snli, 3, int(sys.argv[1]), int(sys.argv[2]), dense_activation=sys.argv[3], batch_size=10000, num_epochs=int(sys.argv[4]))

	if sys.argv[6]=='no':
		c.train()
		c.model.save('../experiments/models/'+sys.argv[7]+'_'+sys.argv[1]+'lstm_'+sys.argv[2]+'dense_10kbatch_'+sys.argv[4]+'epochs.h5')
	else:
		c.load_retrain(sys.argv[5],int(sys.argv[4]))
		c.model.save('../experiments/models/'+sys.argv[7]+'_'+sys.argv[1]+'lstm_'+sys.argv[2]+'dense_10kbatch_retrained_'+str(current_milli_time())+'_'+sys.argv[4]+'epochs.h5')
	c.test()
else:
	print 'Incorrect arguments'