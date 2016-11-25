import data_snli
import w2vec
w2vec = w2vec.W2Vec()
snli = data_snli.SNLI(w2vec)
import concat_lstm
c = concat_lstm.ConcatLstm(snli,3,50,64,batch_size=10000)
c.train()
c.test()
c.model.save('50lstm_64dense_10kbatch.h5')