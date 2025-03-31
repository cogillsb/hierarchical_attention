# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:25:27 2020

@author: cogillsb
"""

import numpy as np
import math
import pandas as pd
import os
import csv
from gensim.models import Word2Vec
import multiprocessing as mp
from sklearn.utils import class_weight
from random import shuffle
from keras.layers import Input, Dense, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras import initializers as initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping



class W2VTrainer:
    '''
    Builds word2vec models from corpi
    '''
    def __init__(self):
        #Model params
        self.w2v_iter = 15
        self.w2v = None
        self.min_ct = 15
        self.embed_len = 100
        self.window_len = 140
        self.w2v_d = None
    
    
    def build_model(self, corpus):
        #Build the model
        self.w2v = Word2Vec(corpus,
        min_count=self.min_ct,
        size=self.embed_len,
        window=self.window_len,
        iter=self.w2v_iter,
        workers=mp.cpu_count())
        
    def build_dict(self):
        self.w2v_d = {wrd:self.w2v[wrd] for wrd in self.w2v.wv.vocab}

            
class Generator:
    '''
    Class to generate matrices
    '''
    def __init__(self, fn, w2v_d):
        self.fn = fn
        self.w2v_d = w2v_d
        self.embed_len = len(self.w2v_d[list(self.w2v_d.keys())[0]])
        self.df_chars = self.desc_csv() 
        self.batch_size = 128     
        self.df_it = None
        self.set_dat()
        
    def set_dat(self):
        #Reading in csv file.
        self.df_it = pd.read_csv(self.fn, iterator=True, chunksize=self.batch_size)
        
    def desc_csv(self):
        #Getting the size and class balances of data
        cnt = 0
        y = []
        with open(self.fn) as f:
            c = csv.reader(f)
            next(c)
            for row in c:
                y.append(row[1])
                cnt += 1
                
        cw = class_weight.compute_class_weight('balanced', np.unique(y), y)
        cw = {0:cw[0], 1:cw[1]}
        
        return {'size':cnt, 'cw':cw}
        
    def embed_x_mat(self, x):
        #Converting data to embeddings for model
        x_mat = []
        for p in x:
            pt = []
            for v in p.split(';'):
                x_vals = v.split(',')
                pt.append([self.w2v_d.get(f, np.zeros(self.embed_len)) for f in x_vals])
            x_mat.append(pt)
        
        return x_mat
    
    def pad_x(self, x):
        #Padding out lengths for a consistent length for lstm.
        for i in range(len(x)):            
            #Pad
            p_len = self.max_len-len(x[i]) +2
            x[i] = np.pad(x[i], ((0, p_len), (0,0), (0,0)), mode='constant')
            
        return x

    def build_x_mat(self, x):
        #Getting matrix for model from text data.
        x = self.embed_x_mat(x)
        x = self.pad_x(x)
        
        return np.array(x)
    
    def get_max_len(self, t):
        #Find the max length of entries for padding.
        mx_len = 0
        for p in t:
            mx_len = max(mx_len, len(p.split(';')))
        self.max_len = mx_len
                
    def __iter__(self):
        return self
    
    def __next__(self):
        
        #Fetch the data
        try:
            df = next(self.df_it)
        except:
            #Reset if iterator runs dry
            self.set_dat()
            df = next(self.df_it)
            
        df = df.reset_index(drop=True)
        df['smpl_wt'] = [self.df_chars['cw'][i] for i in df.Case.values] 
        self.get_max_len(df.val.values)
        x = self.build_x_mat(df.val.values)
     
        return(x, df.Case.values, df.smpl_wt.values)
        
class AttentionWithContext(Layer):
    """
    This is a keras layer that learned a context vector via a dot product with 
    the incoming embeddings.
    """
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        
        a = K.expand_dims(a)
        
        
        weighted_input = x * a
        return [a, K.sum(weighted_input, axis=1)]
       

        
    def compute_output_shape(self, input_shape):
        return [ input_shape, (input_shape[0], input_shape[-1])] 
       

def dot_product(x, kernel):
    """
    Simple keras dot product function moved outside of class to prevent collisions.
    """

    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class DXOModel:
    '''
    Heiarchical attention network LSTM. Architectecture is simple. It runs two layers 
    which learn context vectors. The top layer is a group of the text in the 
    bottom layer. First the context is applied to the bottom layer and then the top layer. 
    Both layers are biderectional LSTMS.
    '''
    def __init__(self):
        self.embed_len = 100

        #How many neurons in the feature recurrent neural net hidden state
        self.bi_rnn1_nrns = 50
        #How many neurons in the visit recurrent neural net hidden state
        self.bi_rnn2_nrns = 200
        #How many neurons in the feature time distributed dense layer
        self.att1_hid = 250
        #How many neurons in the visit time distributed dense layer
        self.att2_hid = 250
        #Dropout rate before the final output.
        self.dropout = 0.4
        
        self.fe = None
        self.model = None
        self.get_field_att = None
        self.get_visit_att = None
        
        
        
    def build_model(self):
        #Enable if gradients go crazy!
        opt =  Adam(clipnorm=1.0)
        #opt = Adam()
        
        #Model for the fields or visit level data
        field_input = Input(shape=(None, self.embed_len,), name='field_input')
        
        #Recurrent layer
        field_rnn = Bidirectional(LSTM(int(self.bi_rnn1_nrns), return_sequences=True, name='lstm1'))(field_input)
     
        #Time distributed dense
        field_dense = TimeDistributed(Dense(int(self.att1_hid)))(field_rnn) 
        
        #Attention
        a1, field_att = AttentionWithContext(name='att1', )(field_dense)
        
        #Return sub heir
        field_encoder = Model(field_input, field_att)
        
        #Model for the visits or patient level data
        visit_input = Input(shape=(None, None, self.embed_len,), name='visit_input')        
        
        #Get first heir
        visit_heir = TimeDistributed(field_encoder)(visit_input)
        
        #Recurrent layer
        visit_rnn = Bidirectional(LSTM(int(self.bi_rnn2_nrns), return_sequences=True, name='lstm2'))(visit_heir)
        
        #Time distributed dense
        visit_dense = TimeDistributed(Dense(int(self.att2_hid)))(visit_rnn)
    
        #Attention layer
        a2, visit_att = AttentionWithContext(name='att2')(visit_dense)
        
        #Dropout
        visit_out = Dropout(self.dropout)(visit_att)
        
        #Output predictions
        preds = Dense(1, activation='sigmoid')(visit_out)
        
        #Build model
        model = Model(visit_input, preds) 
        
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

        self.fe = field_encoder
        self.model = model  
        
class ExportCallback(Callback):
    def __init__(self, out_path, log_name):
        self.out_path = out_path
        self.log_name = log_name

    def on_epoch_end(self, epoch, logs={}):
        os.system('aws s3 cp best_model.h5 %s/best_model.h5' % self.out_path)
        os.system('aws s3 cp %s %s/%s' % (self.log_name, self.out_path, self.log_name))
                  

class Train:
    '''
    Just an examle of how to train the model using various check points 
    for performance plateaus. 
    '''
    def __init__(self, model, train_gen, val_gen):
        self.ptnc = 5
        self.vrbs = 1
        self.epchs = 100
        self.callbacks_list = []
        self. model = model
        self.train_gen = train_gen
        self.train_gen_steps = math.ceil(train_gen.df_chars['size']/train_gen.batch_size)
        self.val_gen = val_gen
        self.val_gen_steps = math.ceil(val_gen.df_chars['size']/val_gen.batch_size)
        self.add_std_callbacks()

        
    def add_std_callbacks(self):
        #Callbacks to stop model and keep best performer
        self.callbacks_list.append(ModelCheckpoint('best_model.h5',
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=True,
                                            mode='auto'))
        self.callbacks_list.append(EarlyStopping(patience=self.ptnc, min_delta=0.0001))
        
    def train(self):
        #Train the model    
        self.model.fit_generator(self.train_gen,
                  validation_data=self.val_gen,
                  steps_per_epoch=self.train_gen_steps,
                  validation_steps=self.val_gen_steps,
                  verbose=self.vrbs,
                  shuffle=True,
                  epochs = self.epchs,
                  callbacks=self.callbacks_list)
        
class TestGenerator(Generator): 
    
    def __next__(self):
        #Fetch the data
        df = next(self.df_it) 
        df = df.reset_index(drop=True)
        self.get_max_len(df.val.values)
        x = self.build_x_mat(df.val.values)
     
        return(df.ID2, x, df.val.values, df.Case.values) 
        
        
class Test:
    """
    Class for loading model and running tests. Importantly,
    this class demonstrates how to pull the relevant weights from the model
    for interpretability.
    Writing out resluts to a csv for some reason. 
    """
    def __init__(self, model, fe, test_gen):
        self.model = model
        self.fe = fe
        self.test_gen = test_gen
        self.test_gen_steps = math.ceil(test_gen.df_chars['size']/test_gen.batch_size)
    
    
    def test_output(self):
        #Attention functions
        get_visit_att = K.function([self.model.get_layer('visit_input').input],
            [self.model.get_layer('att2').output[0]])
        
        get_field_att = K.function([self.fe.get_layer('field_input').input],
                                    [self.fe.get_layer('att1').output[0]])
        
        #Write header
        with open('results.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'True', 'Pred', 'Visit_wts', 'field_wts'])        
       
        #Batch processing
        for _ in range(self.test_gen_steps):
            ids, x, xo, y = next(self.test_gen)
            pred = self.model.predict(x).flatten()
            v_att_wts = get_visit_att([x])[0]
            
            #Patient level  proceesing
            for i in range(len(pred)):
                #Original rec info read back into a matrix format
                rec = []
                for v in xo[i].split(';'):
                    rec.append(v.split(','))

                #flatten visit wts into a str
                vws = ','.join([str(x) for x in v_att_wts[i].flatten()[:len(rec)]])

                #get field wts
                fws = get_field_att([x[i]])[0]
                vfws = {}
                for j in range(len(rec)):
                    vfws[j] = dict(zip(rec[j], fws[j].flatten()))
                
                #Write row
                with open('results.csv','a') as f:
                    writer = csv.writer(f)
                    writer.writerow([ids[i], y[i], pred[i], vws, str(vfws)])
        
