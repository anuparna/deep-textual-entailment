import data_snli
import w2vec
w2vec = w2vec.W2Vec()
snli = data_snli.SNLI(w2vec)

import shared_encoding
c = shared_encoding.SharedEncoding(snli,3,50,64,batch_size=10000)
c.train()
c.model.save('shared_50lstm_64dense_10kbatch.h5')
c.test()
