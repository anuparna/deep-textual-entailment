import data_snli
import w2vec
w2vec = w2vec.W2Vec()
snli = data_snli.SNLI(w2vec)
import concat_lstm
c = concat_lstm.ConcatLstm(snli,3)
c.build_model()
c.train()
c.test()