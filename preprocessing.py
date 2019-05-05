
# -*- coding: utf-8 -*-
import pickle
import data_reader
import re
import numpy as np
from Data import DatasetSchema
import os


class Process(object):
    def __init__(self,opt,train_data_file = 'input/dataset/gap-development.tsv',valid_data_file = 'input/dataset/gap-validation.tsv',test_data_file="input/dataset/gap-test.tsv"):
        self.train_data_file = train_data_file
        self.valid_data_file = valid_data_file
        self.test_data_file = test_data_file
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
    

    def get_train(self, contatenate = 0 ):
    
        tmp = pickle.load(open('4kdoc2label_tokens.pkl','rb'))
        word_index, docs, labels = tmp[0], tmp[1], tmp[2]
        self.opt.word_index = word_index
        print('word_index:',len(word_index))
        print('docs:',len(docs))

        # train data loading
        max_sequence_length =  self.opt.max_sequence_length_contatenate   if contatenate ==1 else self.opt.max_sequence_length
        x_train = data_reader.docs_to_sequences_suffix(docs,word_index,max_sequence_length,contatenate = contatenate )
        y_train = labels # one-hot label encoding
        
        print('[train] Shape of data tensor:', x_train.shape)
        print('[train] Shape of label tensor:', y_train.shape)
        
        self.opt.embedding_matrix = self.build_word_embedding_matrix(word_index)
       # It is better not to build the word embedding matrix  here.

        
        return x_train,y_train
    def get_test(self, contatenate = 0 ): 
        

        # validation data
        valid_temp = pickle.load(open('val_docs2label.pkl','rb'))
        val_docs, val_labels = valid_temp[0],valid_temp[1],
        # sequentializing validation data
        max_sequence_length =  self.opt.max_sequence_length_contatenate   if contatenate ==1 else self.opt.max_sequence_length
        x_val = data_reader.docs_to_sequences_suffix(val_docs,self.opt.word_index ,max_sequence_length , contatenate = contatenate )
        y_val = val_labels # one-hot encoding
        
        print('[Val] Shape of data tensor:', x_val.shape)
        print('[Val] Shape of label tensor:', y_val.shape)

       
        return x_val,y_val

    def get_submit_test(self, contatenate = 0 ): 
        

        # validation data
        tmp_test = pickle.load(open('test2_id2doc.pkl','rb'))
        test_ids, test_docs = tmp_test[0], tmp_test[1]
        # sequentializing validation data
        max_sequence_length =  self.opt.max_sequence_length_contatenate   if contatenate ==1 else self.opt.max_sequence_length
        x_val = data_reader.docs_to_sequences_suffix(test_docs,self.opt.word_index ,max_sequence_length , contatenate = contatenate )
        
        print('[test] Shape of data tensor:', x_val.shape)

       
        return x_val,test_ids

    def get_processed_dataset(self,mode="train"):
        if not os.path.exists("temp"):
            os.mkdir("temp")
        dataset_pkl = "temp/"+self.conf.dataset +"_"+self.conf.split_data+".pkl"
        if os.path.exists(dataset_pkl):
            return pickle.load(open(dataset_pkl, 'rb'))
        if mode == "train":
            filename = os.path.join(self.opt.dataset_dir,"gap-development.tsv")   # waby : gap-development.tsv + gap-test.tsv
            dataset = data_reader.load_data(filename,mode="train")
        elif mode == "dev":
            filename = os.path.join(self.opt.dataset_dir,"gap-validation.tsv")
            dataset = data_reader.load_data(filename,mode="train")
        else:
            filename = os.path.join(self.opt.dataset_dir,"test_stage_2.tsv")
            dataset = data_reader.load_data(filename,mode="test")
            # validation data
            
        # waby : this may not be the right way to rerurn the processed train dataset
        
        test_gene_texts = dataset[:,1]
        ids = dataset[:,2]
        # sequentializing validation data        
        word_index,test_docs = data_reader.tokenizer(test_gene_texts,20000)        
        pickle.dump([ids,test_docs],open('test2_id2doc.pkl', 'wb'))
        return [ids,test_docs]
        
    


