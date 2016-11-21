

class SNLI:
	def __init__(self):
		self.X = {}
		self.y = {}
		self.X['train'],self.y['train'] = self.loadData('train')
		self.X['test'],self.y['test'] = self.loadData('test')
		self.X['dev'],self.y['dev'] = self.loadData('dev')

	def getData(self,settype, onlyGoldLabels=True):
		y = []
		X = []
		with open('../data/snli/snli_1.0_'+settype+'.txt') as datafile:
			prev = None
			for line in datafile:
				if prev is None:
					prev = line
					continue
				parts = line.split("\t")
				if onlyGoldLabels and parts[0] == '-':
					continue
				y.append(parts[0])
				X.append([parts[5],parts[6]])
		return X, y
