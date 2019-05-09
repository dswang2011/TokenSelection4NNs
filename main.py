
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

params = Params()
parser = argparse.ArgumentParser(description='Running Gap.')
parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
args = parser.parse_args()
params.parse_config(args.config)
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score,log_loss


 
def train_model():
    grid_parameters ={
        "cell_type":["lstm","gru","rnn"], 
        "hidden_unit_num":[20,50,75,100,200],
        "dropout_rate" : [0.1,0.2,0.3],#,0.5,0.75,0.8,1]    ,
        "model": ["lstm_2L", "bilstm", "bilstm_2L"],
        "batch_size":[16,32,64],
        "validation_split":[0.05,0.1,0.15,0.2],
        "contatenate":[0,1],
        "lr":[0.001,0.01]       
    }
    # fix cell typ,a nd try different RNN models
    grid_parameters ={
        "cell_type":["gru"], 
        "hidden_unit_num":[50,100],
        "dropout_rate" : [0.3],#,0.5,0.75,0.8,1]    ,
        "model": ["lstm_2L", "bilstm"],
        # "contatenate":[0],
        "lr":[0.001,0.01],
        "batch_size":[32,64],
        # "validation_split":[0.05,0.1,0.15,0.2],
        "validation_split":[0.1],
    }

    token_select = TokenSelection(params)
    train = token_select.get_train(dataset="IMDB",file_name="train.csv",stragety="stopword",POS_category="Noun")
    test = token_select.get_train(dataset="IMDB",file_name="test.csv",stragety="stopword",POS_category="Noun")
#    val_uncontatenated = process.get_test()
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
    for parameter in parameters:
        print(parameter)
        params.setup(zip(grid_parameters.keys(),parameter))        
        model = models.setup(params)
        model.train(train,dev=test)
        


if __name__ == '__main__':
   

    train_model()


    
