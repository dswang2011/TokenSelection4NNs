
# -*- coding: utf-8 -*-
from  Params import Params
import argparse
from preprocessing import Process
import models
from keras.models import load_model
import os
import numpy as np
import itertools
from token_selection import TokenSelection

from sklearn.utils import shuffle

params = Params()
parser = argparse.ArgumentParser(description='Running Gap.')
parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
args = parser.parse_args()
params.parse_config(args.config)
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score,log_loss
import time

def draw_result(predicted, val):
    print('loss:',log_loss(val,predicted)) 
    
    
    ground_label = np.array(val).argmax(axis=1)
    predicted_label = np.array(predicted).argmax(axis=1)
    print('F1:',f1_score(predicted_label ,ground_label,average='macro'))
    print('accuracy:',accuracy_score(predicted_label ,ground_label))
    print(confusion_matrix(predicted_label ,ground_label))

def train_for_document():
    # grid_parameters ={
    #     "cell_type":["lstm","gru","rnn"], 
    #     "hidden_unit_num":[20,50,75,100,200],
    #     "dropout_rate" : [0.1,0.2,0.3],#,0.5,0.75,0.8,1]    ,
    #     "model": ["lstm_2L", "bilstm", "bilstm_2L"],
    #     "batch_size":[16,32,64],
    #     "validation_split":[0.05,0.1,0.15,0.2],
    #     "contatenate":[0,1],
    #     "lr":[0.001,0.01]       
    # }
    # fix cell typ,a nd try different RNN models
    # grid_parameters ={
    #     "cell_type":["gru"], 
    #     "hidden_unit_num":[50],
    #     "dropout_rate" : [0.2],#,0.5,0.75,0.8,1]    ,
    #     "model": [ "bilstm"],
    #     # "contatenate":[0],
    #     "lr":[0.001],
    #     "batch_size":[64],
    #     # "validation_split":[0.05,0.1,0.15,0.2],
    #     "validation_split":[0.1],
    # }
    # CNN parameters
    grid_parameters ={
        "dropout_rate" : [0.3],#,0.5,0.75,0.8,1]    ,
        "model": ["cnn"],
        "filter_size":[30,50],
        "lr":[0.001],
        "batch_size":[64],
        # "validation_split":[0.05,0.1,0.15,0.2],
        "validation_split":[0.1],
    }

    # Set strategy here: strategy = fulltext, stopword, random, POS, dependency, entity ;
    strategy = "fulltext"
    token_select = TokenSelection(params)
    # train,test = token_select.get_train(dataset="IMDB",strategy="entity",selected_ratio=0.5,POS_category="Noun_Verb",cut=2)
    train,test = token_select.get_train(dataset="GAP",strategy=strategy,selected_ratio=0.8,POS_category="Noun_Verb",cut=2)
    # train = token_select.get_train(dataset="IMDB",strategy="entity",selected_ratio=0.8,POS_category="Verb_Adjective",cut=1)
    # X,y = train[0]
    # X,y = shuffle(X, y, random_state=0)
   
#    val_uncontatenated = process.get_test()
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
    for parameter in parameters:
        print(parameter)
        params.setup(zip(grid_parameters.keys(),parameter))   
        model = models.setup(params)
        start = time.time()
        model.train(train,dev=test, strategy=strategy) ### strategy here is just for printing the type
        end = time.time()
        print ("Time spent training .. ", end-start)
def train_for_document_pair():
    # fix cell typ,a nd try different RNN models
    grid_parameters ={
        "cell_type":["gru"], 
        "hidden_unit_num":[200],
        "dropout_rate" : [0.2],#,0.5,0.75,0.8,1]    ,
        "model": ["bilstm"],
        # "contatenate":[0],
        "lr":[0.001],
        "batch_size":[128],
        # "validation_split":[0.05,0.1,0.15,0.2],
        "validation_split":[0.1],
    }
    # grid_parameters ={
    #     "dropout_rate" : [0.3],#,0.5,0.75,0.8,1]    ,
    #     "model": ["cnn"],
    #     "filter_size":[25,50],
    #     "lr":[0.001],
    #     "batch_size":[64],
    #     # "validation_split":[0.05,0.1,0.15,0.2],
    #     "validation_split":[0.1],
    # }
    # Set strategy here:
    strategy = "stopword"
    token_select = TokenSelection(params)
    # process the dataset
    train,test = token_select.get_train(dataset="factcheck",strategy=strategy,POS_category="Verb_Adjective")

#    val_uncontatenated = process.get_test()
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
    for parameter in parameters:
        print(parameter)
        params.setup(zip(grid_parameters.keys(),parameter))        
        model = models.setup(params)
        # here is the invoking of pair function
        model.train_matching(train,dev=test,strategy=strategy)
        # test output
        predicted = model.predict(test[0])
        draw_result(predicted,test[1])

if __name__ == '__main__':
    train_for_document_pair()
    # train_for_document()


    
