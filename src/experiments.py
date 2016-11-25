from data_snli import SNLI
from w2vec import W2Vec

snli_data = SNLI(W2Vec())

from concat_lstm import ConcatLstm
c = ConcatLstm(snli_data,3)
