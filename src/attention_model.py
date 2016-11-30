import re
import json
import argparse
import numpy as np
from collections import Counter
from keras.layers import Input,Embedding,Dropout,LSTM,Dense,merge
from keras.layers.core import Lambda,RepeatVector,Activation,Flatten,Permute,Reshape
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model 
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

judgements={'neutral':0,'entailment':1,'contradiction':2}

def get_H_n(X):
    ans = X[:, -1, :]  # get last element from time dim
    return ans

def get_Y(X, xmaxlen):
    return X[:, :xmaxlen, :]  # get first xmaxlen elem from time dim

def get_R(X):
    Y, alpha = X[0], X[1]
    ans = K.T.batched_dot(Y, alpha)
    return ans

def setParameters():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lstm', action="store", default=150, dest="lstm_units", type=int)
    parser.add_argument('-epochs', action="store", default=30, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=32, dest="batch_size", type=int)
    parser.add_argument('-emb', action="store", default=100, dest="emb", type=int)
    parser.add_argument('-xmaxlen', action="store", default=20, dest="xmaxlen", type=int)
    parser.add_argument('-ymaxlen', action="store", default=20, dest="ymaxlen", type=int)
    parser.add_argument('-maxfeat', action="store", default=35000, dest="max_features", type=int)
    parser.add_argument('-classes', action="store", default=351, dest="num_classes", type=int)
    parser.add_argument('-sample', action="store", default=2, dest="samples", type=int)
    parser.add_argument('-nopad', action="store", default=False, dest="no_padding", type=bool)
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-load', action="store", default=False, dest="load_save", type=bool)
    parser.add_argument('-verbose', action="store", default=False, dest="verbose", type=bool)
    parser.add_argument('-train', action="store", default="train_all.txt", dest="train")
    parser.add_argument('-test', action="store", default="test_all.txt", dest="test")
    parser.add_argument('-dev', action="store", default="dev.txt", dest="dev")
    parameters = parser.parse_args('-train TRAIN'.split())
    print "lstm_units", parameters.lstm_units
    print "epochs", parameters.epochs
    print "batch_size", parameters.batch_size
    print "emb", parameters.emb
    print "samples", parameters.samples
    print "xmaxlen", parameters.xmaxlen
    print "ymaxlen", parameters.ymaxlen
    print "max_features", parameters.max_features
    print "no_padding", parameters.no_padding
    return parameters

def build_model(parameters):
    k = 2 * parameters.lstm_units  # 300
    L = parameters.xmaxlen  # 20
    N = parameters.xmaxlen + parameters.ymaxlen + 1  # for delim
    print "x len", L, "total len", N
    print "k", k, "L", L

    main_input = Input(shape=(N,), dtype='int32', name='main_input')
    x = Embedding(output_dim=parameters.emb, input_dim=parameters.max_features, input_length=N, name='x')(main_input)
    drop_out = Dropout(0.1, name='dropout')(x)
    lstm_fwd = LSTM(parameters.lstm_units, return_sequences=True, name='lstm_fwd')(drop_out)
    lstm_bwd = LSTM(parameters.lstm_units, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
    bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')
    drop_out = Dropout(0.1)(bilstm)
    h_n = Lambda(get_H_n, output_shape=(k,), name="h_n")(drop_out)
    Y = Lambda(get_Y, arguments={"xmaxlen": L}, name="Y", output_shape=(L, k))(drop_out)
    Whn = Dense(k, W_regularizer=l2(0.01), name="Wh_n")(h_n)
    Whn_x_e = RepeatVector(L, name="Wh_n_x_e")(Whn)
    WY = TimeDistributed(Dense(k, W_regularizer=l2(0.01)), name="WY")(Y)
    merged = merge([Whn_x_e, WY], name="merged", mode='sum')
    M = Activation('tanh', name="M")(merged)

    alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
    flat_alpha = Flatten(name="flat_alpha")(alpha_)
    alpha = Dense(L, activation='softmax', name="alpha")(flat_alpha)

    Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)

    r_ = merge([Y_trans, alpha], output_shape=(k, 1), name="r_", mode=get_R)

    r = Reshape((k,), name="r")(r_)

    Wr = Dense(k, W_regularizer=l2(0.01))(r)
    Wh = Dense(k, W_regularizer=l2(0.01))(h_n)
    merged = merge([Wr, Wh], mode='sum')
    h_star = Activation('tanh')(merged)
    out = Dense(3, activation='softmax')(h_star)
    output = out
    model = Model(input=[main_input], output=output)
    print main_input
    model.summary()
    #plot(model, 'model.png')
    # # model.compile(loss={'output':'binary_crossentropy'}, optimizer=Adam())
    # model.compile(loss={'output':'categorical_crossentropy'}, optimizer=Adam(options.lr))
    model.compile(loss='categorical_crossentropy',optimizer=Adam(parameters.lr),metrics=['accuracy'])
    return model

def getWordsFromSentence(sentence):
    return [word.strip().lower() for word in re.split('(\W+)?', sentence) if word.strip()]

def createVocab(sentence1_words,sentence2_words): #Arguments are premise and hypothesis
    vocab=Counter()
    all_words=sentence1_words+sentence2_words #Concat words
    vocab.update(all_words) #contains the count of all the words
    list_of_words = ["unk", "delimiter"] + [ word for word, count in vocab.iteritems() if count > 0]
    vocab = dict([(word,enum_num) for enum_num,word in enumerate(list_of_words)])
    return vocab

def concatPremiseHypothesis(premise, hypothesis, delim_val):
    num_examples = premise.shape[0]  
    glue = delim_val * np.ones(num_examples).reshape(num_examples, 1)
    print(glue)
    print(premise)
    print(hypothesis)
    concat_string = np.concatenate((premise, glue, hypothesis), axis=1)
    print concat_string
    return concat_string

def compute_accuracy(premise, hypothesis, model, parameters):
    scores = model.predict(premise)#, batch_size=parameters.batch_size)
    prediction = np.zeros(scores.shape)
    for i in range(scores.shape[0]):
        l = np.argmax(scores[i])
        prediction[i][l] = 1.0
    assert np.array_equal(np.ones(prediction.shape[0]), np.sum(prediction, axis=1))
    plabels = np.argmax(prediction, axis=1)
    tlabels = np.argmax(hypothesis, axis=1)
    acc = accuracy(tlabels, plabels)
    return acc, acc

def loadTrainingDataFromJson(parameters):
    vocab=Counter()
    final_input=list()
    p_list=list()
    h_list=list()
    delim_list=list()
    judgements_train=list()
    max_length_sentence = parameters.xmaxlen
    counter=0
    for l in open('D:\Study !\Info_Retrieval\project\Text Entailment\snli_1.0\snli_1.0\snli_1.0_train.jsonl', 'r'):
        parsed_json = json.loads(l)
        #Hypothesis
        sentence1=parsed_json["sentence1"]
        print(sentence1)
        sentence1_words=getWordsFromSentence(sentence1)
        #Premise
        sentence2=parsed_json["sentence2"]
        print(sentence2)
        sentence2_words=getWordsFromSentence(sentence2)
        #Create a vocab using both sentences
        wordsDict=createVocab(sentence1_words,sentence2_words)
        delim_list.append(wordsDict["delimiter"])
        #Segregate Training data
        p_list.append([wordsDict[word] if word in wordsDict else 0 for word in sentence1_words])
        h_list.append([wordsDict[word] if word in wordsDict else 0 for word in sentence2_words])
        judgements_train+=[judgements[parsed_json["gold_label"]]]
        
        #Currently uses only only 2 samples just to create the plots
        counter+=1        
        if counter==2:
            break
    judgements_train = to_categorical(judgements_train, nb_classes=3)
    p_list = pad_sequences(p_list, maxlen=max_length_sentence, value=0, padding='pre')
    h_list = pad_sequences(h_list, maxlen=max_length_sentence, value=0, padding='post') 
    final_input= concatPremiseHypothesis(p_list, h_list, 1)    
    return judgements_train,final_input

if __name__ == "__main__":
    parameters=setParameters()
    model=build_model(parameters)
    
    judgements_train,final_input = loadTrainingDataFromJson(parameters)
    #print premise_train.shape, hypothesis_train.shape, judgements_train.shape, concatednated_train.shape
    
    # Fit the model
    hist = model.fit(final_input,judgements_train,
                           batch_size=parameters.batch_size,
                           nb_epoch=parameters.epochs)
    
    
    # print(hist.history)
    model.save('D:/my_model.h5')
    #print compute_accuracy(premise_train, hypothesis_train, model, parameters)