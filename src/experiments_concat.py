import data_snli
import w2vec

w2vec = w2vec.W2Vec()
snli = data_snli.SNLI(w2vec)

import sys
import concat_lstm

if len(sys.argv) == 3:
	print 'lstmsize:',sys.argv[1]
	print 'densesize:',sys.argv[2]
	c = concat_lstm.ConcatLstm(snli, 3, int(sys.argv[1]), int(sys.argv[2]), batch_size=10000)
	c.train()
	c.model.save('../experiments/models/concat_'+sys.argv[1]+'lstm_'+sys.argv[2]+'dense_10kbatch.h5')
	c.test()
else:
	print 'Incorrect arguments'