import data_snli
import w2vec
w2vec = w2vec.W2Vec()
snli = data_snli.SNLI(w2vec)

import shared_encoding

for lstmsize in [10,20,30,50]:
	for densesize in [20, 32, 64, 128]:
		c = shared_encoding.SharedEncoding(snli, 3, lstmsize, densesize, batch_size=10000)
		c.train()
		c.model.save('../experiments/models/shared_'+str(lstmsize)+'lstm_'+str(densesize)+'dense_10kbatch.h5')
		c.test()