import data_snli
import w2vec

w2vec = w2vec.W2Vec()
snli = data_snli.SNLI(w2vec)

import sys
import shared_encoding

if len(sys.argv) == 3:
	print 'lstmsize:',sys.argv[1]
	print 'densesize:',sys.argv[2]
	c = shared_encoding.SharedEncoding(snli, 3, int(sys.argv[1]), int(sys.argv[2]), batch_size=10000)
	c.train()
	c.model.save('../experiments/models/shared_'+sys.argv[1]+'lstm_'+sys.argv[2]+'dense_10kbatch.h5')
	c.test()
else:
	print 'Incorrect arguments'