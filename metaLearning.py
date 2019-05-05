from sklearn.linear_model import LogisticRegression
from preprocessing import Process
from  Params import Params
import argparse
import os
from keras.models import load_model
import numpy as np
import itertools
from numpy import argmax
from sklearn.preprocessing import LabelEncoder

params = Params()
parser = argparse.ArgumentParser(description='Running Gap.')
parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
args = parser.parse_args()
params.parse_config(args.config)
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score,log_loss

label_encoder = LabelEncoder()

def emsemble(vals,dir_name="selected_model"):
    predicts= [] 
   
    count = 0    
    for filename in os.listdir( dir_name ):
        if not filename.startswith("best"):
            if len(filename.split("_"))<3:
                continue
            if count>10:
                break
            concate_str = filename.split("_")[-3]
            if 'contatenate-' not in concate_str:
                concate_str = filename.split("_")[-2]
            contatenate_flag= int(concate_str[-1])
            # print('concate_flag:',contatenate_flag)
            val = vals[contatenate_flag]
            model_file = os.path.join(dir_name,filename)
            model = load_model(model_file)
            predicted = model.predict(val[0])
            
            # if log_loss(val[1], predicted)<0.73:
            #     print(filename + ": " + str(log_loss(val[1], predicted)))
            #     count+=1
            # else:
            #     continue
            predicted = [argmax(i) for i in predicted]
            predicts.append(predicted)
    results = np.transpose(np.asarray(predicts))
    print(results.shape)
    return results

def draw_result(predicted, val):   
    # ground_label = np.array(val).argmax(axis=1)
    # predicted_label = np.array(predicted).argmax(axis=1)
    print('F1:',f1_score(predicted ,val,average='macro'))
    print('accuracy:',accuracy_score(predicted ,val))
    print(confusion_matrix(predicted ,val))
    try:
        print('loss:',log_loss(val,predicted)) 
    except:
        print('loss print error')

def logist_regression(results,labels):


    process = Process(params)
    train_uncontatenated = process.get_train()
    train_contatenated = process.get_train(contatenate =1)
    predicted_arr = emsemble([train_uncontatenated,train_contatenated]) 
    inverted = [argmax(i) for i in train_uncontatenated[1]]
    print('shape',predicted_arr.shape)
    print(np.asarray(inverted).shape)
    

    for c in [0.001,0.1,0.25,1]:
        for penalty in ['l2']:
            clf = LogisticRegression(C=c, random_state=0, solver='lbfgs',penalty = penalty, multi_class='multinomial')
            # print('coef_:',clf.coef_)
            clf.fit(predicted_arr,inverted)
            predicted = clf.predict(results)
            draw_result(predicted, labels)
    # print('loss:',log_loss(val,predicted)) 
    


def test_model():
    process = Process(params)
    _ = process.get_train()  #waby: in orde to get get the test set properly [build word index parameter], actually.
    val_uncontatenated = process.get_test()
#    train_contatenated = process.get_train(contatenate =1)
    val_contatenated = process.get_test(contatenate =1)
    results = emsemble([val_uncontatenated,val_contatenated])    
    inverted = [argmax(i) for i in val_contatenated[1]]
    print('valid data shape:',results.shape)
    logist_regression(results,inverted)
    # inverted = label_encoder.inverse_transform((val_contatenated[1])) 
    
    # draw_result(predicted,inverted)



if __name__ == '__main__':
	test_model()