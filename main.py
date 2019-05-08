
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


 


def emsemble(vals,dir_name="selected_model"):
    predicts= [] 
   
    
    for filename in os.listdir( dir_name ):
        if not filename.startswith("best") and '0.6490612626075745_BiLSTM2L' in filename:
            if len(filename.split("_"))<3:
                continue
            concate_str = filename.split("_")[-3]
            if 'contatenate-' not in concate_str:
                concate_str = filename.split("_")[-2]
            contatenate_flag= int(concate_str[-1])
            print('concate_flag:',contatenate_flag)
            val = vals[contatenate_flag]
            model_file = os.path.join(dir_name,filename)
            model = load_model(model_file)
            predicted = model.predict(val[0])
            # print(filename + ": " + str(log_loss(val[1], predicted)))
            predicts.append(predicted)
    return np.mean(predicts,axis=0)


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
    # CNN parameters
    # grid_parameters ={
    #     "dropout_rate" : [0.3],#,0.5,0.75,0.8,1]    ,
    #     "model": ["cnn"],
    #     "filter_size":[30],
    #     "contatenate":[0,1],
    #     "lr":[0.001,0.01],
    #     "batch_size":[32,64],
    #     # "validation_split":[0.05,0.1,0.15,0.2],
    #     "validation_split":[0.10,0.15,0.2],
    # }
    # grid_parameters ={
    #     "cell_type":["gru"], 
    #     "hidden_unit_num":[20],
    #     "dropout_rate" : [0.3],#,0.5,0.75,0.8,1],
    #     "model": ["bilstm_2L"],
    #     "contatenate":[0,1],
    #     "lr":[0.01],
    #     "batch_size":[32],
    #     "validation_split":[0.1],
    # }
    token_select = TokenSelection(params)
    train = token_select.get_train(dataset="IMDB",file_name="train.csv",stragety="stopword",POS_category="Noun")
#    val_uncontatenated = process.get_test()
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
    for parameter in parameters:
        print(parameter)
        params.setup(zip(grid_parameters.keys(),parameter))        
        model = models.setup(params)
        model.train(train)
        

def draw_result(predicted, val):
    print('loss:',log_loss(val,predicted)) 
    
    
    ground_label = np.array(val).argmax(axis=1)
    predicted_label = np.array(predicted).argmax(axis=1)
    print('F1:',f1_score(predicted_label ,ground_label,average='macro'))
    print('accuracy:',accuracy_score(predicted_label ,ground_label))
    print(confusion_matrix(predicted_label ,ground_label))
   

def test_model():
    process = Process(params)
    _ = process.get_train()  #waby: in orde to get get the test set properly [build word index parameter], actually.
    val_uncontatenated = process.get_test()
#    train_contatenated = process.get_train(contatenate =1)
    val_contatenated = process.get_test(contatenate =1)
    predicted = emsemble([val_uncontatenated,val_contatenated])    
    

    draw_result(predicted,val_contatenated[1])
    
#    predicted = emsemble(train)  
#    print(log_loss(train[1],predicted))
#    draw_result(predicted,train)
def output_submit(output_path):
	process = Process(params)
	_ = process.get_train()  #waby: in orde to get get the test set properly [build word index parameter], actually.

	# load test data
	test_uncontatenated = process.get_submit_test()
	test_contatenated = process.get_submit_test(contatenate =1)
	y_prediction = emsemble([test_uncontatenated,test_contatenated]) 
	print("ID,A,B,NEITHER")
	fw = open(output_path,'a')
	fw.write('ID,A,B,NEITHER\n')
	ids = test_contatenated[1]
	for i in range(len(y_prediction)):
		print(ids[i],','.join([str(x) for x in y_prediction[i]]), sep=',', file=fw)

if __name__ == '__main__':
   
#    test_model()
    train_model()
    # test_model()
    # output_submit('output_ds.txt')

    
