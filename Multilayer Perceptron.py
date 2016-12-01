
# coding: utf-8

# In[1]:

import sklearn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import string
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
from random import shuffle


# In[2]:

# read the entire file into a python array
with open('data/snli/snli_1.0_train.jsonl', 'rb') as f:
    data = f.readlines()

# remove the trailing "\n" from each line
data = map(lambda x: x.rstrip(), data)

# each element of 'data' is an individual JSON object.
# i want to convert it into an *array* of JSON objects
# which, in and of itself, is one large JSON object
# basically... add square brackets to the beginning
# and end, and have all the individual business JSON objects
# separated by a comma
data_json_str = "[" + ','.join(data) + "]"

# now, load it into pandas
train_df = pd.read_json(data_json_str)


# In[3]:

train_df.head()


# In[4]:

train_df = train_df[train_df['gold_label'] != "-"]


# In[7]:

with open('data/snli/snli_1.0_test.jsonl', 'rb') as f:
    data = f.readlines()

# remove the trailing "\n" from each line
data = map(lambda x: x.rstrip(), data)

# each element of 'data' is an individual JSON object.
# i want to convert it into an *array* of JSON objects
# which, in and of itself, is one large JSON object
# basically... add square brackets to the beginning
# and end, and have all the individual business JSON objects
# separated by a comma
data_json_str = "[" + ','.join(data) + "]"

# now, load it into pandas
test_df = pd.read_json(data_json_str)


# In[8]:

test_df = test_df[test_df['gold_label'] != "-"]


# In[9]:

tr_df = train_df[["sentence1","sentence2","gold_label"]]


# In[10]:

tr_df["input"] = tr_df["sentence1"].map(str) + tr_df["sentence2"]


# In[11]:

tr_df.head()


# In[12]:

te_df = test_df[["sentence1","sentence2","gold_label"]]


# In[13]:

te_df["input"] = te_df["sentence1"].map(str) + te_df["sentence2"]


# In[14]:

vectorizer = CountVectorizer(ngram_range=(1, 2),min_df=1,max_features=6000)
X_train = vectorizer.fit_transform(tr_df["input"])


# In[15]:

X_train.shape


# In[16]:

le = preprocessing.LabelEncoder()
le.fit(["entailment", "neutral", "contradiction"])
Y_train = le.transform(tr_df["gold_label"]) 


# In[17]:

Y_train.shape


# In[18]:

vectorizer1 = CountVectorizer(ngram_range=(1, 2),min_df=1,max_features=6000)
X_test = vectorizer1.fit_transform(te_df["input"])
X_test.shape


# In[55]:

le1 = preprocessing.LabelEncoder()
le1.fit(["entailment", "neutral", "contradiction"])
Y_test = le.transform(te_df["gold_label"]) 
Y_test.shape


# In[21]:

clf = MLPClassifier(solver='lbgfs', alpha=1e-5,
                     hidden_layer_sizes=(200,200), random_state=1)


# In[22]:

clf.fit(X_train, Y_train)


# In[58]:

Y_pred = clf.predict(X_test)


# In[59]:

print accuracy_score(Y_test,Y_pred)


# In[61]:

tr_df['input']


# In[76]:

tr_df['input1'] = tr_df.apply (lambda row: row['input'].encode('ascii').translate(None, string.punctuation).lower(),axis=1)


# In[77]:

tr_df['input1'][0]


# In[78]:

te_df['input1'] = te_df.apply (lambda row: row['input'].encode('ascii').translate(None, string.punctuation).lower(),axis=1)


# In[86]:

f1 = open('data/train-entailment.txt','w')
f2 = open('data/train-contradiction.txt','w')
f3 = open('data/train-neutral.txt','w')
for index, row in tr_df.iterrows():
    if (row['gold_label'] == "entailment"):
        f1.write(row['input1'] + '\n')
    elif (row['gold_label'] == "contradiction"):
        f2.write(row['input1'] + '\n')
    else:
        f3.write(row['input1'] + '\n')    
f1.close() 
f2.close() 
f3.close() 


# In[87]:

f1 = open('data/test-entailment.txt','w')
f2 = open('data/test-contradiction.txt','w')
f3 = open('data/test-neutral.txt','w')
for index, row in te_df.iterrows():
    if (row['gold_label'] == "entailment"):
        f1.write(row['input1'] + '\n')
    elif (row['gold_label'] == "contradiction"):
        f2.write(row['input1'] + '\n')
    else:
        f3.write(row['input1'] + '\n')    
f1.close() 
f2.close() 
f3.close() 


# In[88]:

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


# In[95]:

sources = {'test-entailment.txt':'TEST_ENT', 'test-contradiction.txt':'TEST_CON', 'test-neutral.txt':'TEST_NTL', 
           'train-entailment.txt':'TRAIN_ENT', 'train-contradiction.txt':'TRAIN_CON', 'train-neutral.txt':'TRAIN_NTL'}


# In[96]:

sentences = LabeledLineSentence(sources)


# In[97]:

model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=2)


# In[98]:

model.build_vocab(sentences.to_array())


# In[99]:

for epoch in range(30):
    model.train(sentences.sentences_perm())


# In[102]:

model.most_similar('happy')


# In[122]:

train_arrays = numpy.zeros((549367, 100))
train_labels = numpy.zeros(549367)


# In[112]:

tr_df.groupby('gold_label').count()


# In[123]:

for i in range(183187):
    prefix_train_con = 'TRAIN_CON_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_con]
    #train_arrays[12500 + i] = model[prefix_train_neg]
    train_labels[i] = 1


# In[124]:

for i in range(183416):
    prefix_train_ent = 'TRAIN_ENT_' + str(i)
    train_arrays[183187 + i] = model.docvecs[prefix_train_ent]
    #train_arrays[12500 + i] = model[prefix_train_neg]
    train_labels[183187 + i] = 2


# In[125]:

for i in range(182764):
    prefix_train_ntl = 'TRAIN_NTL_' + str(i)
    train_arrays[183187 + 183416 + i] = model.docvecs[prefix_train_ntl]
    #train_arrays[12500 + i] = model[prefix_train_neg]
    train_labels[183187 + 183416 + i] = 3


# In[126]:

test_arrays = numpy.zeros((9824, 100))
test_labels = numpy.zeros(9824)


# In[119]:

te_df.groupby('gold_label').count()


# In[127]:

for i in range(3237):
    prefix_test_con = 'TEST_CON_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_con]
    #train_arrays[12500 + i] = model[prefix_train_neg]
    test_labels[i] = 1


# In[128]:

for i in range(3368):
    prefix_test_ent = 'TEST_ENT_' + str(i)
    test_arrays[3237 + i] = model.docvecs[prefix_test_ent]
    #train_arrays[12500 + i] = model[prefix_train_neg]
    test_labels[3237 + i] = 2


# In[129]:

for i in range(3219):
    prefix_test_ntl = 'TEST_NTL_' + str(i)
    test_arrays[3368 + 3237 + i] = model.docvecs[prefix_test_ntl]
    #train_arrays[12500 + i] = model[prefix_train_neg]
    test_labels[3368 + 3237 + i] = 3


# In[130]:

clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(200, 200), random_state=1)


# In[131]:

clf1.fit(train_arrays,train_labels)


# In[132]:

predicted_labels = clf1.predict(test_arrays)


# In[133]:

print accuracy_score(test_labels,predicted_labels)


# In[ ]:



