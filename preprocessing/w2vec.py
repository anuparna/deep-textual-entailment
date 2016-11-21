from gensim.models import Word2Vec
import numpy as np 

#depends on pip packages: pattern, gensim, numpy
class W2Vec:
	def __init__(self):
		print 'loading'
		self.model = Word2Vec.load_word2vec_format('../data/w2vec/GoogleNews-vectors-negative300.bin', binary=True)
		print 'loaded'

	def convertWord(self, word, unk='random'):
		return self.model[word]

	def convertSentence(self, sentence):
		wordsreps = []
		for w in sentence:
			wordsreps.append(self.convertWord(w))

# w = W2Vec()
# print w.convertWord('the')