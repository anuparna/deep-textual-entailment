import data_snli
import w2vec
w2vec = w2vec.W2Vec()
snli = data_snli.SNLI(w2vec)

import concat_lstm

for lstmsize in [10,20,30,50]:
	for densesize in [20, 32, 64, 128]:
		if lstmsize==10 and densesize==20:
			continue
		c = concat_lstm.ConcatLstm(snli, 3, lstmsize, densesize, batch_size=10000)
		c.train()
		c.model.save('../experiments/models/concat_'+str(lstmsize)+'lstm_'+str(densesize)+'dense_10kbatch.h5')
		c.test()
