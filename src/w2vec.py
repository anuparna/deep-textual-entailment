from gensim.models import Word2Vec
import numpy as np 

#depends on pip packages: pattern, gensim, numpy
class W2Vec:
	def __init__(self):
		print 'Loading word2vec model'
		self.model = Word2Vec.load_word2vec_format('../data/w2vec/GoogleNews-vectors-negative300.bin', binary=True)
		print 'Loaded Word2Vec model'

	def convertWord(self, word, unk='random'):
		return self.model[word]

	def convertSentence(self, sentence):
		return self.model[sentence]

	def unkWordRep(self):
		return np.random.rand(300)