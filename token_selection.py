import pickle  
from  Params import Params
import argparse
from stanfordcorenlp import StanfordCoreNLP
import CoreNLP 
import data_reader
import os
from keras.utils import to_categorical
import numpy as np

class TokenSelection(object):
    def __init__(self,opt):
        self.opt=opt   

    def build_word_embedding_matrix(self,word_index):
        # word embedding lodading
        embeddings_index = data_reader.get_embedding_dict(self.opt.glove_dir)
        print('Total %s word vectors.' % len(embeddings_index))

        # initial: random initial (not zero initial)
        embedding_matrix = np.random.random((len(word_index) + 1,self.opt.embedding_dim  ))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    # stragety = fulltext, stopword, random, POS, dependency ;
    def get_train(self,dataset,file_name,stragety="random",selected_ratio=0.9,cut=1,POS_category="Noun"):
        tokens_list = []
        labels = []
        if stragety=="POS":
            Noun,Verb,Adjective,Noun_Verb,Noun_Adjective,Verb_Adjective,Noun_Verb_Adjective,labels = self.get_selected_token(dataset,file_name,stragety,selected_ratio,cut)
            if POS_category=="Noun":
                tokens_list=np.array(Noun)
            elif POS_category=="Verb":
                tokens_list=np.array(Verb)
            elif POS_category=="Adjective":
                tokens_list=np.array(Adjective)
            elif POS_category=="Noun_Verb":
                tokens_list=np.array(Noun_Verb)
            elif POS_category=="Noun_Adjective":
                tokens_list=np.array(Noun_Adjective)
            elif POS_category=="Verb_Adjective":
                tokens_list=np.array(Verb_Adjective)
            elif POS_category=="Noun_Verb_Adjective":
                tokens_list=np.array(Noun_Verb_Adjective)
        else:
            tokens_list,labels = self.get_selected_token(dataset,file_name,stragety,selected_ratio,cut) 
        max_sequence_length=self.opt.max_sequence_length
        # max_num_words = self.opt.max_num_words
        word_index = data_reader.tokenizer(tokens_list,self.opt.max_nb_words)
        self.opt.word_index = word_index
        print('word_index:',len(word_index))

        # padding
        x_train = data_reader.tokens_list_to_sequences(tokens_list,word_index,max_sequence_length)
        y_train = to_categorical(np.asarray(labels)) # one-hot encoding y_train = labels # one-hot label encoding
        
        print('[train] Shape of data tensor:', x_train.shape)
        print('[train] Shape of label tensor:', y_train.shape)
        self.opt.embedding_matrix = self.build_word_embedding_matrix(word_index)
        return x_train,y_train

    # stragety = fulltext, stopword, random, POS, dependency;
    def get_selected_token(self,dataset,file_name,stragety="random",selected_ratio=0.9,cut=1):
        output_root = "prepared/"+dataset+"/"
        if stragety == "fulltext":
        	fulltext_pkl = output_root+file_name+"_fulltext.pkl"
        	temp = pickle.load(open(fulltext_pkl,'rb'))
        	token_lists,labels = temp[0],temp[1]
        	return token_lists,labels
        elif stragety == "stopword":
            stopword_pkl = output_root+file_name+"_stopword.pkl"
            temp = pickle.load(open(stopword_pkl,'rb'))
            token_lists,labels = temp[0],temp[1]
            return token_lists,labels
        elif stragety =="random":
            random_pkl = output_root+file_name+"_random"+str(selected_ratio)+".pkl"
            temp = pickle.load(open(random_pkl,'rb'))
            token_lists,labels = temp[0],temp[1]
            return token_lists,labels
        elif stragety =="POS":
            pos_pkl = output_root+file_name+"_pos.pkl"
            temp = pickle.load(open(pos_pkl,'rb'))
            Noun,Verb,Adjective,Noun_Verb,Noun_Adjective,Verb_Adjective,Noun_Verb_Adjective,labels = temp[0],temp[1],temp[2],temp[3],temp[4],temp[5],temp[6],temp[7]
            return Noun,Verb,Adjective,Noun_Verb,Noun_Adjective,Verb_Adjective,Noun_Verb_Adjective,labels
        elif stragety =="dependency":
            dependency_pkl = output_root+file_name+"_treecut"+str(cut)+".pkl"
            temp = pickle.load(open(dependency_pkl,'rb'))
            token_lists,labels = temp[0],temp[1]
            return token_lists,labels

    # If the data is prepared/IMDB/train.txt => dataset=IMDB, file_name=train.txt
    def token_selection_preparation(self,nlp,dataset,file_name):
        output_root = "prepared/"+dataset+"/"
        if not os.path.exists(output_root):
            os.mkdir(output_root)
        # load data
        file_path = os.path.join(output_root,file_name) 
        texts,labels = data_reader.load_classification_data(file_path,hasHead=0)    # set with 1 if there is head

        # full text
        fulltext_pkl = output_root+file_name+"_fulltext.pkl"
        if not os.path.exists(fulltext_pkl):
            tokens_list = CoreNLP.text2tokens_fulltext(nlp,texts)
            pickle.dump([tokens_list,labels],open(fulltext_pkl, 'wb'))
            print('output succees:',fulltext_pkl)
            print('shape:',tokens_list.shape)
        else:
            print("Already exist:",fulltext_pkl)

        # stodword
        stopword_pkl = output_root+file_name+"_stopword.pkl"
        if not os.path.exists(stopword_pkl):
            tokens_list = CoreNLP.text2tokens_stopword(nlp,texts)
            pickle.dump([tokens_list,labels],open(stopword_pkl, 'wb'))
            print('output succees:',stopword_pkl)
            print('shape:',tokens_list.shape)
        else:
            print("Already exist:",stopword_pkl)
        # random
        for selected_ratio in [0.9,0.8,0.7,0.6,0.5]:
            random_pkl = output_root+file_name+"_random"+str(selected_ratio)+".pkl"
            if not os.path.exists(random_pkl):
                tokens_list = CoreNLP.text2tokens_random(nlp,texts,selected_ratio)
                pickle.dump([tokens_list,labels],open(random_pkl, 'wb'))
                print('output succees:',random_pkl)
                print('shape:',tokens_list.shape)
            else:
                print("Already exists:",random_pkl)
        # POS combination
        pos_pkl = output_root+file_name+"_pos.pkl"
        if not os.path.exists(pos_pkl):
            Noun,Verb,Adjective,Noun_Verb,Noun_Adjective,Verb_Adjective,Noun_Verb_Adjective = CoreNLP.text2token_POS(nlp,texts)
            pickle.dump([Noun,Verb,Adjective,Noun_Verb,Noun_Adjective,Verb_Adjective,Noun_Verb_Adjective,labels],open(pos_pkl, 'wb'))
            print('output succees (POSs):',pos_pkl)
            print('each with shape:',np.array(Noun).shape)
        else:
            print("Already exist:",pos_pkl)
            # print('each with shape:',np.array(Noun).shape)

        # dependency tree different cuts
        cuts = [1,2]
        tokens_dict_list = CoreNLP.text2tokens_treecuts(nlp,texts,cuts)
        for cut in cuts:
            dependency_pkl = output_root+file_name+"_treecut"+str(cut)+".pkl"
            if not os.path.exists(dependency_pkl):
                tokens_list=[]
                for tokens_dict in tokens_dict_list:
                    tokens_list.append(tokens_dict[cut]) 
                pickle.dump([tokens_list,labels],open(dependency_pkl, 'wb'))
                print('output succees:',dependency_pkl)
                print('shape:',tokens_list.shape)
            else:
                print("Already exists:",dependency_pkl)
        
# prepare the tokens list into pkl files.
if __name__ == '__main__':
    params = Params()
    parser = argparse.ArgumentParser(description='Set configuration file.')
    parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
    parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
    parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
    args = parser.parse_args()
    params.parse_config(args.config)

    token_select = TokenSelection(params)
    nlp = StanfordCoreNLP(params.corenlp_root)
    # below is where you need to set your data name
    token_select.token_selection_preparation(nlp = nlp, dataset="IMDB",file_name="train.csv")
    token_select.token_selection_preparation(nlp = nlp, dataset="IMDB",file_name="test.csv")
    nlp.close() # Do not forget to close! The backend server will consume a lot memery.

