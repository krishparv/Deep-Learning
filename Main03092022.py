import csv
import enum
import os
import sys
from datetime import datetime
#import tensorflow
import tensorflow as tf
#from tensorflow import keras
import keras
import numpy as np
import unicodedata
from keras.layers import GlobalMaxPooling1D
import pyxdameraulevenshtein

#from datasetcreator import damerau_levenshtein, jaccard, jaro, jaro_winkler,monge_elkan, cosine, strike_a_match, soft_jaccard, sorted_winkler, permuted_winkler, skipgram, davies


from tempfile import mkdtemp
import os.path as path

#os.environ['KERAS_BACKEND'] = 'theano'
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, Bidirectional,  Layer, Lambda, Permute
from keras.layers.core import Masking
from keras import backend as K, Model
from keras import initializers
#from keras import initializations

from keras.layers import dot

class GlobalMaxPooling1DMasked(GlobalMaxPooling1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(GlobalMaxPooling1DMasked, self).__init__(**kwargs)
    def build(self, input_shape): super(GlobalMaxPooling1DMasked, self).build(input_shape)
    def call(self, x, mask=None): return super(GlobalMaxPooling1DMasked, self).call(x)

def damerau_levenshtein ( str1 , str2 ):
    aux = pyxdameraulevenshtein.normalized_damerau_levenshtein_distance( str1 , str2 )
    return 1.0 - aux

class SelfAttLayer(Layer):
    def __init__(self, **kwargs):
        self.attention = None
        self.init = initializations.get('normal')
        self.supports_masking = True
        super(SelfAttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.init((input_shape[-1],))
        self.trainable_weights = [self.W]
        super(SelfAttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1))
        ai = K.exp(eij)
        weights = ai/K.expand_dims(K.sum(ai, axis=1),1)
        weighted_input = x*K.expand_dims(weights,2)
        self.attention = weights
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape): return (input_shape[0], input_shape[-1])
    def compute_output_shape(self, input_shape): return self.get_output_shape_for(input_shape)

def AlignmentAttention(input_1, input_2):
    def unchanged_shape(input_shape): return input_shape
    def softmax(x, axis=-1):
        ndim = K.ndim(x)
        if ndim == 2: return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            return e / s
        else: raise ValueError('Cannot apply softmax to a tensor that is 1D')
    w_att_1 = Sequential()
    w_att_1.add(dot([input_1, input_2], axes=-1, normalize=False))
    w_att_1.add(Lambda(lambda x: softmax(x, axis=1), output_shape=unchanged_shape))
    w_att_2 = Sequential()
    w_att_2.add(dot([input_1, input_2], axes=-1, normalize=False))
    w_att_2.add(Lambda(lambda x: softmax(x, axis=2), output_shape=unchanged_shape))
    w_att_2.add(Permute((2, 1)))
    in1_aligned = Sequential()
    in1_aligned.add(dot([w_att_1, input_1], axes=1, normalize=False))
    in2_aligned = Sequential()
    in2_aligned.add(dot([w_att_2, input_2], axes=1, normalize=False))
    q1_combined = Sequential()
    #q1_combined.add(Merge([input_1,in2_aligned], mode='concat'))
    q1_concat = Concatenate([input_1.output, in2_aligned.output], axis=-1)
    q1_combined.add(Model(inputs=[input_1.input, in2_aligned.input], outputs=q1_concat))
    q2_combined = Sequential()
    q2_concat = Concatenate([input_2.output, in1_aligned.output], axis=-1)
    #q2_combined.add(Merge([input_2,in1_aligned], mode='concat'))
    q2_combined.add(Model(inputs=[input_2.input, in1_aligned.input], outputs=q2_concat))
    return q1_combined, q2_combined


def deep_neural_net_gru(train_data_1, train_data_2, train_labels, test_data_1, test_data_2, test_labels, max_len,
                        len_chars, hidden_units=60, bidirectional=True, selfattention=True, maxpooling=False,
                        alignment=True, shortcut=True, multiplerlu=True, onlyconcat=False, n=1):
    early_stop = EarlyStopping(monitor='loss', patience=0, verbose=1)
    checkpointer = ModelCheckpoint(filepath="checkpoint" + str(n) + ".hdf5", verbose=1, save_best_only=True)
   # gru1 = GRU(hidden_units, consume_less='gpu', return_sequences=True)
    #gru2 = GRU(hidden_units, consume_less='gpu', return_sequences=(alignment or #selfattention or maxpooling))
    gru1 = GRU(hidden_units,  return_sequences=True)
    gru2 = GRU(hidden_units, return_sequences=(alignment or selfattention or maxpooling))
	if bidirectional:
        gru1 = Bidirectional(gru1)
        gru2 = Bidirectional(gru2)
    # definition for left branch of the network
    left_branch = Sequential()
    left_branch.add(Masking(mask_value=0, input_shape=(max_len, len_chars)))
    #if error in this shortcut condition, set shortcut =false
	if shortcut:
        left_branch_aux1 = Sequential()
        left_branch_aux1.add(left_branch)
        left_branch_aux1.add(gru1)
        left_branch_aux2 = Sequential()
        # left_branch_aux2.add(Merge([left_branch, left_branch_aux1], mode='concat'))
       # merged1 = keras.layers.concatenate([left_branch.output, left_branch_aux1.output])
       # left_branch_aux2.add(merged1)
        #left_branch_aux2 = Model(inputs=[left_branch.input, left_branch_aux1.input], outputs=merged1)
        #left_branch = left_branch_aux2.add(merged1)merged1 = Concatenate([left_branch.output, left_branch_aux1.output], axis=-1)

        left_branch_aux2 = Model(inputs=[left_branch.input, left_branch_aux1.input], outputs=merged1)
        left_branch = left_branch_aux2
    else:
        left_branch.add(gru1)
    left_branch.add(Dropout(0.01))
    left_branch.add(gru2)
    left_branch.add(Dropout(0.01))
    # definition for right branch of the network
    right_branch = Sequential()
    right_branch.add(Masking(mask_value=0, input_shape=(max_len, len_chars)))
    if shortcut:
        right_branch_aux1 = Sequential()
        right_branch_aux1.add(right_branch)
        right_branch_aux1.add(gru1)
        right_branch_aux2 = Sequential()
        merged2 = Concatenate([right_branch.output, right_branch_aux1.output], axis=-1)

        right_branch_aux2 = Model(inputs=[right_branch.input, right_branch_aux1.input], outputs=merged2)

		#merged2 = keras.layers.concatenate([right_branch.output, right_branch_aux1.output], axis=-1)
        #right_branch_aux2 = Model(inputs=[right_branch.input, right_branch_aux1.input], outputs=merged2)

        # right_branch_aux2.add(Merge() ([right_branch, right_branch_aux1], mode='concat'))
        #right_branch = right_branch_aux2
    else:
        right_branch.add(gru1)
    right_branch.add(Dropout(0.01))
    right_branch.add(gru2)
    right_branch.add(Dropout(0.01))
    # mechanisms used for building representations from the GRU states (e.g., through attention)
    if alignment: left_branch, right_branch = AlignmentAttention(left_branch, right_branch)
    #if error in this SelfAttLayer condition, set selfattention=false or comment it.
	if selfattention:
        att = SelfAttLayer()
        left_branch.add(att)
        right_branch.add(att)
    elif maxpooling:
        left_branch.add(GlobalMaxPooling1DMasked())
        right_branch.add(GlobalMaxPooling1DMasked())
    elif alignment:
        gru3 = GRU(hidden_units, consume_less='gpu', return_sequences=False)
        if bidirectional: gru3 = Bidirectional(gru3)
        left_branch.add(gru3)
        right_branch.add(gru3)
    
	# combine the two representations and produce the final classification
    con_layer = Sequential(name="con_layer")
    merge_con = keras.layers.concatenate([left_branch.output, right_branch.output], axis=-1)
    # con_layer.add(Merge([left_branch, right_branch], mode='concat', name="merge_con"))
    con_layer = Model([left_branch.input, right_branch.input], outputs=merge_con,name="con_model")
    
	mul_layer = Sequential(name="mul_layer")

    # mul_layer.add(Merge([left_branch, right_branch], mode='mul', name="merge_mul"))
    multiply_layer = keras.layers.Multiply()
    merge_mul = multiply_layer([left_branch.output, right_branch.output])
    mul_layer = Model([left_branch.input, right_branch.input], outputs=merge_mul, name="mul_model")
	
	dif_layer = Sequential(name="dif_layer")
    #v2
	#dif_layer.add(Merge([left_branch, right_branch],
     #                   mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0], name="merge_dif"))
    
	# if error in lambda,use subtract
	dif_layer.add(Lambda(lambda x: x[0]-x[1], output_shape=lambda x: x[0])([left_branch, right_branch]))
    #merge_dif=keras.layers.Subtract()([left_branch.output,right_branch.output])
    #dif_layer=Model([left_branch.input, right_branch.input], outputs=merge_dif,name="dif_model")
	
	final_model = Sequential(name="final_model")
    if onlyconcat:
		#v2
        #final_model.add(con_layer)
		merge_threeconcat=con_layer		
    else:
        #v2
        #final_model.add(keras.layers.Add()(merge_threeconcat))
		#final_model.add(Merge([con_layer, mul_layer, dif_layer], mode='concat', name="merge_threeconcat"))
		merge_threeconcat = keras.layers.concatenate([con_layer.output, mul_layer.output, dif_layer.output], axis=-1)
    final_model.add(Dropout(0.01))
    final_model.add(Dense(hidden_units, activation='relu'))
    final_model.add(Dropout(0.01))
    if multiplerlu:
        final_model.add(Highway(activation='relu'))
        final_model.add(Dropout(0.01))
        final_model.add(Highway(activation='relu'))
        final_model.add(Dropout(0.01))
    final_model.add(Dense(1, activation='sigmoid'))
    #if error in above lines,comment above 8 lines and uncomment below one line 
	dense=Dense(hidden_units,activation='relu')(merge_threeconcat)
    output=Dense(1,activation='sigmoid')(dense)
	final_model=Model([left_branch.input, right_branch.input],output,name="final_model")
	final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    final_model.fit([train_data_1, train_data_2], train_labels,
                    validation_data=([test_data_1, test_data_2], test_labels),
                    callbacks=[early_stop, checkpointer], nb_epoch=20)
    #start_time = time.time()
    start_time = datetime.now()
	#v2
	#aux = final_model.predict_classes([test_data_1, test_data_2]).ravel()
    aux = final_model.predict([test_data_1, test_data_2])
    #return aux, (time.time() - start_time)
	return aux, (datetime.now() - start_time).total_seconds()*3600

def evaluate_deep_neural_net(dataset='dataset-string-similarity.txt', method='gru', training_instances=-1,
                             bidirectional=True, hiddenunits=60):
    max_seq_len = 40
    num_true = 0.0
    num_false = 0.0
    num_true_predicted_true = 0.0
    num_true_predicted_false = 0.0
    num_false_predicted_true = 0.0
    num_false_predicted_false = 0.0
    timer = 0.0
    with open(dataset) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2"], delimiter='|')
        for row in reader:
            if row['res'] == "TRUE":
                num_true += 1.0
            else:
                num_false += 1.0
    XA1 = []
    XB1 = []
    XC1 = []
    Y1 = []
    XA2 = []
    XB2 = []
    XC2 = []
    Y2 = []
    start_time = datetime.now()
    print('Reading dataset... ' + str(datetime.now() - start_time))
    with open(dataset) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2"], delimiter='|')
        start_time = datetime.now()
        for row in reader:
            if row['res'] == "TRUE":  # append first 5 to Y1 then to Y2
                if len(Y1) < ((num_true + num_false) / 2.0):
                    Y1.append(1)  # 5 true
                else:
                    Y2.append(1)  # 5 true
            else:
                if len(Y1) < ((num_true + num_false) / 2.0):
                    Y1.append(0)
                else:
                    Y2.append(0)
            # assign first words in row['s1'] & second in row['s2']
            #row['s1'] = row['s1'].decode('utf-8')
            row['s1'] = row['s1']
            if row['s2'] is not None:
                #row['s2'] = row['s2'].decode('utf-8')
                row['s2'] = row['s2']

			#split s1 and s2 by space and get count of words present
        
            #get string having highest count of word
            #assign highest one to array a1 
            #if(c1>c2):
            
            
            c1=row['s1'].split('+')
            c2=row['s2'].split('+')
            if(row['s1'].count('+')>=row['s2'].count('+')):
                arr1=row['s1'].split('+')
                arr2=row['s2'].split('+')
            else:
               arr1=row['s2'].split('+')
               arr2=row['s1'].split('+')
            #compare arr2 string with arr1 using string matching method

            mat=np.zeros((len(arr2),len(arr1))) 
            for i,ar2 in enumerate(arr2):
                for j,ar1 in enumerate(arr1):
                    mat[i,j]=damerau_levenshtein(ar1,ar2)

            arrnew1=arr1.copy()
            
            for i, ar in enumerate(arrnew1):
                arrnew1[i]='0'

            s11=0
            ps=0
            #for i,ar2 in enumerate(arr2):
            for j,mat1 in enumerate(mat):
                s11=0
                ps=-1
                for k,m in enumerate(mat1):
                    if(m>0.2 and m>=s11):
                        s11=m
                        ps=k
                if(ps!=-1):
                    arrnew1[ps]=arr2[j]        

            print(arrnew1)
            if(arrnew1.count('0')==len(arrnew1)):
                continue

            row['s1']='+'.join(arr1)
            row['s2']='+'.join(arrnew1)
			#arrange arr2 with string in  highest value location

            row['s1'] = bytearray(unicodedata.normalize('NFKD', (u'|' + row['s1'] + u'|')), encoding='utf-8')
            if row['s2'] is not None:
                row['s2'] = bytearray(unicodedata.normalize('NFKD', (u'|' + row['s2'] + u'|')), encoding='utf-8')
            else:
                row['s2'] = "test"

            # asssign normalized byte array of s1 and s2 into XA1 and XB1 as training data
            # asssign normalized byte array of s1 and s2 into XA2 and XB2 as input data
            if len(XA1) < ((num_true + num_false) / 2.0):
                XA1.append(row['s1'])
                XB1.append(row['s2'])
            else:
                XA2.append(row['s1'])
                XB2.append(row['s2'])
        print("Dataset read... " + str(datetime.now() - start_time))
    Y1 = np.array(Y1, dtype=np.bool)
    Y2 = np.array(Y2, dtype=np.bool)
    chars = list(set(list([val for sublist in XA1 + XB1 + XA2 + XB2 for val in sublist])))
    char_labels = {ch: i for i, ch in enumerate(chars)}
    aux1 = np.memmap("temporary-file-dnn-1-" + method, mode="w+", shape=(len(XA1), max_seq_len, len(chars)),
                     dtype=np.bool)
    for i, example in enumerate(XA1):
        for t, char in enumerate(example):
            if t < max_seq_len:
                aux1[i, t, char_labels[char]] = 1
            else:
                break
    XA1 = aux1
    aux1 = np.memmap("temporary-file-dnn-2-" + method, mode="w+", shape=(len(XB1), max_seq_len, len(chars)),
                     dtype=np.bool)
    for i, example in enumerate(XB1):
        for t, char in enumerate(example):
            if t < max_seq_len:
                aux1[i, t, char_labels[char]] = 1
            else:
                break
    XB1 = aux1
    aux1 = np.memmap("temporary-file-dnn-3-" + method, mode="w+", shape=(len(XA2), max_seq_len, len(chars)),
                     dtype=np.bool)
    for i, example in enumerate(XA2):
        for t, char in enumerate(example):
            if t < max_seq_len:
                aux1[i, t, char_labels[char]] = 1
            else:
                break
    XA2 = aux1
    aux1 = np.memmap("temporary-file-dnn-4-" + method, mode="w+", shape=(len(XB2), max_seq_len, len(chars)),
                     dtype=np.bool)
    for i, example in enumerate(XB2):
        for t, char in enumerate(example):
            if t < max_seq_len:
                aux1[i, t, char_labels[char]] = 1
            else:
                break
    XB2 = aux1
    print("Temporary files created... " + str(datetime.now() - start_time))
    print("Training classifiers...")

    if training_instances <= 0: training_instances = min(len(Y1), len(Y2))

    aux1, time1 = deep_neural_net_gru(train_data_1=XA1[0:training_instances, :, :],
                                      train_data_2=XB1[0:training_instances, :, :],
                                      train_labels=Y1[0:training_instances, ], test_data_1=XA2, test_data_2=XB2,
                                      test_labels=Y2, max_len=max_seq_len, len_chars=len(chars),
                                      bidirectional=bidirectional, hidden_units=hiddenunits, n=1)
    aux2, time2 = deep_neural_net_gru(train_data_1=XA2[0:training_instances, :, :],
                                      train_data_2=XB2[0:training_instances, :, :],
                                      train_labels=Y2[0:training_instances, ], test_data_1=XA1, test_data_2=XB1,
                                      test_labels=Y1, max_len=max_seq_len, len_chars=len(chars),
                                      bidirectional=bidirectional, hidden_units=hiddenunits, n=2)

    timer += time1 + time2

   # print("Matching records")
    #real = list(Y1) + list(Y2)
    file = open("dataset-dnn-accuracy", "w+")

    num_true_predicted_true += 1.0
    num_true_predicted_false += 1.0
    num_false_predicted_false += 1.0
    print(str(num_true_predicted_true) + ":" + str(num_false_predicted_true))
    timer = (timer / float(int(num_true + num_false))) * 50000.0
    acc = (num_true_predicted_true + num_false_predicted_false) / (num_true + num_false)
    pre = (num_true_predicted_true) / (num_true_predicted_true + num_false_predicted_true)
    rec = (num_true_predicted_true) / (num_true_predicted_true + num_true_predicted_false)
    f1 = 2.0 * ((pre * rec) / (pre + rec))
    file.close()
    print("Metric = Deep Neural Net Classifier :", method.upper())
    print("Bidirectional :", bidirectional)
    print("Accuracy =", acc)
    print("Precision =", pre)
    print("Recall =", rec)
    print("F1 =", f1)
    print("Processing time per 50K records =", timer)
    print("Number of training instances =", training_instances)
    print("")
    os.remove("temporary-file-dnn-1-" + method)
    os.remove("temporary-file-dnn-2-" + method)
    os.remove("temporary-file-dnn-3-" + method)
    os.remove("temporary-file-dnn-4-" + method)
    sys.stdout.flush()


evaluate_deep_neural_net(dataset="dataset_final_jrc_person.csv")
