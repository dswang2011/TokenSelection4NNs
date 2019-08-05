
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
import pprint
import util

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



def grid_search_parameters(grid_parameters, strategy, train, test, dict_results, dataset):
    parameters = [arg for index, arg in enumerate(itertools.product(*grid_parameters.values())) if
                  index % args.gpu_num == args.gpu]
    val_acc = 0
    max_acc = 0
    local_time=0.0
    for parameter in parameters:
        print(parameter)
        params.setup(zip(grid_parameters.keys(), parameter))
        model = models.setup(params)
        if dataset in params.pair_set.split(","):
            val_acc, time_spent, model = model.train_matching(train, dev=test, strategy=strategy,dataset=dataset)  ### strategy here is just for printing the type
        else:    
            val_acc, time_spent, model = model.train(train, dev=test, strategy=strategy,dataset=dataset)  ### strategy here is just for printing the type
        if float(val_acc) > max_acc:
            max_acc=float(val_acc)
            local_time=time_spent
        if dataset not in dict_results:
            dict_results[dataset] = {}
        if model not in dict_results[dataset]:
            dict_results[dataset][model] = {}
        if strategy not in dict_results[dataset][model]:
            dict_results[dataset][model][strategy] = {"val_acc":val_acc, "time":time_spent}

        if val_acc > dict_results[dataset][model][strategy]['val_acc']:
            dict_results[dataset][model][strategy]['val_acc'] = val_acc
            dict_results[dataset][model][strategy]['time'] = time_spent
    return model,max_acc,local_time


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
    # grid_parameters =
    # {
    #     "cell_type":["gru"],
    #     "hidden_unit_num":[50],
    #     "dropout_rate" : [0.2],#,0.5,0.75,0.8,1]    ,
    #     "model": [ "bilstm"],
    #     # "contatenate":[0],
    #     "lr":[0.001],
    #     "batch_size":[32,64],
    #     # "validation_split":[0.05,0.1,0.15,0.2],
    #     "validation_split":[0.1],
    # }

    models = [
    	#CNN parameters
    	{
        "dropout_rate" : [0.3],#,0.5,0.75,0.8,1]    ,
        "model": ["cnn"],
        # "filter_size":[30],
        "filter_size":[30,50],
        "lr":[0.1,0.0001],
        # "batch_size":[32],
        "batch_size":[32,64],
        # "validation_split":[0.05,0.1,0.15,0.2],
        "validation_split":[0.1],
        },
    	# RNN parameters
        {
            "cell_type": ["gru"],
            "hidden_unit_num": [100],
            "dropout_rate": [0.2],  # ,0.5,0.75,0.8,1]    ,
            "model": ["bilstm_2L"],
            # "contatenate":[0],
            "lr": [0.1,0.0001],
            "batch_size": [32,64],
            # "validation_split":[0.05,0.1,0.15,0.2],
            "validation_split": [0.1],
        }
    ]
    # grid_parameters ={
    #     "dropout_rate" : [0.3],#,0.5,0.75,0.8,1]    ,
    #     "model": ["cnn"],
    #     "filter_size":[30],
    #     # "filter_size":[30,50],
    #     "lr":[0.001],
    #     "batch_size":[32],
    #     # "batch_size":[32,64],
    #     # "validation_split":[0.05,0.1,0.15,0.2],
    #     "validation_split":[0.1],
    # }

    file_summary_results = open("summary_results_2.txt", "a")
    file_local = "local_results_2.txt"

    dict_results = {}
    datasets = ["factcheck"]
    for dataset in datasets:
        for grid_parameters in models:
            # Set strategy here: strategy = fulltext, stopword, random, POS, dependency, entity ;
            #if strategy="POS", then POS_category works, possible value: "Noun", "Verb", "Adjective", "Noun_Verb", "Noun_Adjective", "Verb_Adjective", "Noun_Verb_Adjective".
            #
            # POS_categories = ["Noun", "Verb", "Adjective", "Noun_Verb", "Noun_Adjective", "Verb_Adjective", "Noun_Verb_Adjective"]
            POS_categories = ["Noun_Verb","Noun_Adjective", "Verb_Adjective", "Noun_Verb_Adjective"]
            selected_ratios = [0.9,0.8,0.7,0.6]
            # selected_ratios = [0.8]
            cuts = [1,2]
            sig_num = [3,4,5,6,7,8]

            dict_strategies = {
                                "fulltext": {},
                                # "stopword": {},
                                # "random": {},
                                # "POS":{},
                               # "dependency":{},
                               # "entity":{},
                               # "IDF":{},
                               # "IDF_blocks":{},
                               # "IDF_blocks_pos":{}	# sig_num = [3,4,5,6,7]
                               }

            for strategy in dict_strategies:

                if strategy == "fulltext" or strategy == "stopword" or strategy == "entity" or strategy == "triple":
                    token_select = TokenSelection(params)
                    train,test = token_select.get_train(dataset=dataset,strategy=strategy)
                    print('running:',strategy)
                    model,max_acc,loc_time = grid_search_parameters(grid_parameters, strategy, train, test, dict_results, dataset)
                    util.write_line(dataset+'-'+model+'-'+strategy+'->'+str(max_acc)+'\t'+str(loc_time)+'s',file_local)
                elif strategy == "POS":
                    for pos_cat in POS_categories:
                        token_select = TokenSelection(params)
                        train,test = token_select.get_train(dataset=dataset,strategy=strategy, POS_category=pos_cat)
                        print('running:',strategy,':',pos_cat)
                        model,max_acc,loc_time = grid_search_parameters(grid_parameters, strategy, train, test, dict_results, dataset)
                        util.write_line(dataset+'-'+model+'-'+strategy+'-'+pos_cat+'->'+str(max_acc)+'\t'+str(loc_time)+'s',file_local)
                elif strategy == "random" or strategy == "IDF":
                    for ratio in selected_ratios:
                        token_select = TokenSelection(params)
                        train, test = token_select.get_train(dataset=dataset, strategy=strategy, selected_ratio=ratio)
                        print('running:',strategy,':',ratio)
                        model,max_acc,loc_time=grid_search_parameters(grid_parameters, strategy, train, test, dict_results, dataset)
                        util.write_line(dataset+'-'+model+'-'+strategy+'-'+str(ratio)+'->'+str(max_acc)+'\t'+str(loc_time)+'s', file_local)
                elif strategy == "dependency":
                    for cut in cuts:
                        token_select = TokenSelection(params)
                        train, test = token_select.get_train(dataset=dataset, strategy=strategy, cut=cut)
                        model,max_acc,loc_time=grid_search_parameters(grid_parameters, strategy, train, test, dict_results, dataset)
                        util.write_line(dataset+'-'+model+'-'+strategy+'-'+str(cut)+'->'+str(max_acc)+'\t'+str(loc_time)+'s', file_local)
                elif strategy == "IDF_blocks" or strategy == "IDF_blocks_pos":
                    for sig_n in sig_num:
                        token_select = TokenSelection(params)
                        train, test = token_select.get_train(dataset=dataset, strategy=strategy, sig_num=sig_n)
                        model,max_acc,loc_time=grid_search_parameters(grid_parameters, strategy, train, test, dict_results, dataset)
                        util.write_line(dataset+'-'+model+'-'+strategy+'-'+str(sig_n)+'->'+str(max_acc)+'\t'+str(loc_time)+'s', file_local)
                else:
                    pprint('========WRONG strategy=================')
                    util.write_line('====wrong strategy======'+dataset+'-'+strategy,file_local)
                    pprint('PLESE check!!!==================')
    pprint.pprint(dict_results)
    pprint.pprint(dict_results, file_summary_results)
    file_summary_results.close()


if __name__ == '__main__':
    # train_for_document_pair()
    train_for_document()


    
