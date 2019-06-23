import pickle  
from  Params import Params
import argparse
from stanfordcorenlp import StanfordCoreNLP
import CoreNLP 
import data_reader
import os
from keras.utils import to_categorical
import numpy as np
from sklearn import preprocessing
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

	# strategy = fulltext, stopword, random, POS, dependency , IDF, 
	def get_train(self,dataset,strategy="fulltext",selected_ratio=0.9,cut=1,POS_category="Noun",sig_num=3):
		print('=====strategy:',strategy,'pos_cat:',POS_category,' cut:',cut,'======')
		tokens_list_train_test = []
		labels_train_test = []
		for file_name in ["train.csv","test.csv"]:
			tokens_list,labels = self.get_selected_token(dataset,file_name,strategy,selected_ratio=selected_ratio,cut=cut)
			tokens_list_train_test.append(tokens_list)
			labels_train_test.append(labels)
		self.opt.nb_classes = len(set(labels))
		print(set(labels))
		# max_num_words = self.opt.max_num_words
		if dataset in self.opt.pair_set.split(","):
			all_tokens= [set(sentence) for tokens_list,tokens_list1 in tokens_list_train_test for sentence in tokens_list1]
		else:
			all_tokens= [set(sentence) for dataset in tokens_list_train_test for sentence in dataset]
		word_index = data_reader.tokenizer(all_tokens,self.opt.max_nb_words)
		self.opt.word_index = word_index
		print('word_index:',len(word_index))

		le = preprocessing.LabelEncoder()
		# labels = le.fit_transform(labels)
		# print(labels)
		# padding
		train_test = []
		for tokens_list,labels in zip(tokens_list_train_test,labels_train_test):
			if dataset in self.opt.pair_set.split(","):
				x1 = data_reader.tokens_list_to_sequences(tokens_list[0],word_index,self.opt.max_sequence_length)
				x2 = data_reader.tokens_list_to_sequences(tokens_list[1],word_index,self.opt.max_sequence_length)
				x = [x1,x2]
			else:
				x = data_reader.tokens_list_to_sequences(tokens_list,word_index,self.opt.max_sequence_length)
			y = le.fit_transform(labels)
			# print(y)
			y = to_categorical(np.asarray(y)) # one-hot encoding y_train = labels # one-hot label encoding
			train_test.append([x,y])
			if dataset in self.opt.pair_set.split(","):
				print('[train pair] Shape of data tensor:', x[0].shape,' and ', x[1].shape)
			else:
				print('[train] Shape of data tensor:', x.shape)
			print('[train] Shape of label tensor:', y.shape)
		self.opt.embedding_matrix = self.build_word_embedding_matrix(word_index)
		return train_test

	# strategy = fulltext, stopword, random, POS, dependency;
	def get_selected_token(self,dataset,file_name,strategy,selected_ratio=0.9,cut=1,sig_num=3,POS_category="Noun"):
		customized_tokens = ['aaac','bbbc','pppc','pppcs']
		output_root = "prepared/"+dataset+"/"
		if strategy == "fulltext":
			pkl_file_path = output_root+file_name+"_fulltext.pkl"
		elif strategy == "triple":
			pkl_file_path = output_root+file_name+"_triple.pkl"
		elif strategy == "stopword":
			pkl_file_path = output_root+file_name+"_stopword.pkl"
		elif strategy =="random":
			pkl_file_path = output_root+file_name+"_random"+str(selected_ratio)+".pkl"
		elif strategy =="POS":
			pkl_file_path = output_root+file_name+"_pos.pkl"
			temp = pickle.load(open(pkl_file_path,'rb'))
			if dataset in self.opt.pair_set.split(","):
				Noun,Verb,Adjective,Noun_Verb,Noun_Adjective,Verb_Adjective,Noun_Verb_Adjective, \
				Noun1,Verb1,Adjective1,Noun_Verb1,Noun_Adjective1,Verb_Adjective1,Noun_Verb_Adjective1,labels = \
					temp[0],temp[1],temp[2],temp[3],temp[4],temp[5],temp[6],\
					temp[7],temp[8],temp[9],temp[10],temp[11],temp[12],temp[13],temp[14]
				if POS_category=="Noun":
					tokens_list=np.array([Noun,Noun1])
				elif POS_category=="Verb":
					tokens_list=np.array([Verb,Verb1])
				elif POS_category=="Adjective":
					tokens_list=np.array([Adjective,Adjective1])
				elif POS_category=="Noun_Verb":
					tokens_list=np.array([Noun_Verb,Noun_Verb1])
				elif POS_category=="Noun_Adjective":
					tokens_list=np.array([Noun_Adjective,Noun_Adjective1])
				elif POS_category=="Verb_Adjective":
					tokens_list=np.array([Verb_Adjective,Verb_Adjective1])
				elif POS_category=="Noun_Verb_Adjective":
					tokens_list=np.array([Noun_Verb_Adjective,Noun_Verb_Adjective1])
				return tokens_list,labels
			else:
				Noun,Verb,Adjective,Noun_Verb,Noun_Adjective,Verb_Adjective,Noun_Verb_Adjective,labels = \
					temp[0],temp[1],temp[2],temp[3],temp[4],temp[5],temp[6],temp[7]
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
				return tokens_list,labels
		elif strategy =="dependency":
			pkl_file_path = output_root+file_name+"_treecut"+str(cut)+".pkl"
		elif strategy == "entity":
			pkl_file_path = output_root+file_name+"_entity.pkl"
		elif strategy=="IDF":
			idf_pkl = output_root+"train.csv"+"_idf.pkl"
			self.idf_dict = pickle.load(open(idf_pkl,'rb'))
			if dataset in self.opt.pair_set.split(","):
				texts1,texts2,labels = data_reader.load_data_overall(dataset,file_name)
				tokens_list1=CoreNLP.text2tokens_idf(texts1,self.idf_dict,customized_tokens,select_ratio=selected_ratio)
				tokens_list2=CoreNLP.text2tokens_idf(texts2,self.idf_dict,customized_tokens,select_ratio=selected_ratio)
				return [tokens_list1,tokens_list2],labels
			else:
				texts,labels = data_reader.load_data_overall(dataset,file_name)
				tokens_list=CoreNLP.text2tokens_idf(texts,self.idf_dict,customized_tokens,select_ratio=selected_ratio)
				return tokens_list,labels
		elif strategy=="IDF_blocks":
			pkl_file_path = output_root+file_name+"_idf_block"+str(sig_num)+".pkl"
			temp = pickle.load(open(pkl_file_path,'rb'))
			if dataset in self.opt.pair_set.split(","):
				tokens_list,tokens_list1,tokens_list_pos,tokens_list_pos1,labels = temp[0],temp[1],temp[2],temp[3],temp[4]
				return [tokens_list,tokens_list1],labels
			else:
				tokens_list,tokens_list_pos,labels = temp[0],temp[1],temp[2]
				return tokens_list,labels
		elif strategy=="IDF_blocks_pos":
			pkl_file_path = output_root+file_name+"_idf_block"+str(sig_num)+".pkl"
			temp = pickle.load(open(pkl_file_path,'rb'))
			if dataset in self.opt.pair_set.split(","):
				tokens_list,tokens_list1,tokens_list_pos,tokens_list_pos1,labels = temp[0],temp[1],temp[2],temp[3],temp[4]
				return [tokens_list_pos,tokens_list_pos1],labels
			else:
				tokens_list,tokens_list_pos,labels = temp[0],temp[1],temp[2]
				return tokens_list_pos,labels

		temp = pickle.load(open(pkl_file_path,'rb'))
		tokens_list,labels = temp[0],temp[1]
		return tokens_list,labels

	# If the data is prepared/IMDB/train.txt => dataset=IMDB, file_name=train.txt
	def token_selection_preparation(self,nlp,dataset,file_name):
		output_root = "prepared/"+dataset+"/"
		print(output_root)
		if not os.path.exists(output_root):
			os.mkdir(output_root)
		# load data
		file_path = os.path.join(output_root,file_name)
		print('load data:',file_path)
		if dataset in self.opt.pair_set.split(","):
			texts,texts2,labels = data_reader.load_data_overall(dataset,file_name,test100=False)	# set with 1 if there is head
			print('=== this is a paired text dataset ===')
		else:
			texts,labels = data_reader.load_data_overall(dataset,file_name)	# set with 1 if there is head
		# customized 
		customized_tokens = ['aaac','bbbc','pppc','pppcs']
		# full text
		fulltext_pkl = output_root+file_name+"_fulltext.pkl"
		if not os.path.exists(fulltext_pkl):
			tokens_list = CoreNLP.text2tokens_fulltext(nlp,texts)
			
			if dataset in self.opt.pair_set.split(","):
				tokens_list1 = CoreNLP.text2tokens_fulltext(nlp,texts2)
				pickle.dump([[tokens_list,tokens_list1],labels],open(fulltext_pkl, 'wb'))
			else:
				pickle.dump([tokens_list,labels],open(fulltext_pkl, 'wb'))
			print('output succees:',fulltext_pkl)
			print('shape:',tokens_list.shape)
		else:
			print("Already exist:",fulltext_pkl)

		# load triples
		# triple_pkl = output_root+file_name+"_triple.pkl"
		# if not os.path.exists(triple_pkl):
		#	 triple_path = os.path.join(output_root,file_name.replace('.csv','_triples.txt'))
		#	 triple_texts,labels = data_reader.load_triple_data(triple_path)
		#	 tokens_list = CoreNLP.text2tokens_fulltext(nlp,triple_texts)
		#	 pickle.dump([tokens_list,labels],open(triple_pkl, 'wb'))
		#	 print('output success',triple_pkl)
		#	 print('shape:',tokens_list.shape)
		# else:
		#	 print('Alredy exists:',triple_pkl)

		# stodword
		stopword_pkl = output_root+file_name+"_stopword.pkl"
		if not os.path.exists(stopword_pkl):
			tokens_list = CoreNLP.text2tokens_stopword(nlp,texts)
			
			if dataset in self.opt.pair_set.split(","):
				tokens_list1 = CoreNLP.text2tokens_stopword(nlp,texts2)
				pickle.dump([[tokens_list,tokens_list1],labels],open(stopword_pkl, 'wb'))
			else:
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
				if dataset in self.opt.pair_set.split(","):
					tokens_list1 = CoreNLP.text2tokens_random(nlp,texts2,selected_ratio)
					pickle.dump([[tokens_list,tokens_list1],labels],open(random_pkl, 'wb'))
				else:
					pickle.dump([tokens_list,labels],open(random_pkl, 'wb'))
				print('output succees:',random_pkl)
				print('shape:',tokens_list.shape)
			else:
				print("Already exists:",random_pkl)
		# POS combination
		pos_pkl = output_root+file_name+"_pos.pkl"
		if not os.path.exists(pos_pkl):
			Noun,Verb,Adjective,Noun_Verb,Noun_Adjective,Verb_Adjective,Noun_Verb_Adjective = CoreNLP.text2token_POS(nlp,texts,customized_tokens)
			if dataset in self.opt.pair_set.split(","):
				Noun1,Verb1,Adjective1,Noun_Verb1,Noun_Adjective1,Verb_Adjective1,Noun_Verb_Adjective1 = \
					CoreNLP.text2token_POS(nlp,texts2,customized_tokens)
				# 1-> 8 (0->7)
				pickle.dump([Noun,Verb,Adjective,Noun_Verb,Noun_Adjective,Verb_Adjective,Noun_Verb_Adjective,\
					Noun1,Verb1,Adjective1,Noun_Verb1,Noun_Adjective1,Verb_Adjective1,Noun_Verb_Adjective1,labels],open(pos_pkl, 'wb'))
			else:
				pickle.dump([Noun,Verb,Adjective,Noun_Verb,Noun_Adjective,Verb_Adjective,Noun_Verb_Adjective,labels],open(pos_pkl, 'wb'))
			print('output succees (POSs):',pos_pkl)
			print('each with shape:',np.array(Noun).shape)
		else:
			print("Already exist:",pos_pkl)
			# print('each with shape:',np.array(Noun).shape)

#dependency tree different cuts
		cuts = [1,2,3]
		exists=[]
		for cut in cuts:
			dependency_pkl = output_root+file_name+"_treecut"+str(cut)+".pkl"
			if os.path.exists(dependency_pkl):
				print("Already exists:",dependency_pkl)
				exists.append(cut)
		cuts = [x for x in cuts if x not in exists]
		# excute for the remaining
		if len(cuts)>0:
			tokens_dict_list = CoreNLP.text2tokens_treecuts(nlp,texts,cuts)
			if dataset in self.opt.pair_set.split(","):
				tokens_dict_list1 = CoreNLP.text2tokens_treecuts(nlp,texts,cuts)
				for cut in cuts:
					dependency_pkl = output_root+file_name+"_treecut"+str(cut)+".pkl"
					if not os.path.exists(dependency_pkl):
						tokens_list,tokens_list1=[],[]
						for tokens_dict in tokens_dict_list:
							tokens_list.append(tokens_dict[cut])
						for tokens_dict1 in tokens_dict_list1:
							tokens_list1.append(tokens_dict1[cut]) 
						pickle.dump([[tokens_list,tokens_list1],labels],open(dependency_pkl, 'wb'))
						print('output succees:',dependency_pkl)
						print('shape:',np.array(tokens_list).shape)
					else:
						print("Already exists:",dependency_pkl)
			else:
				for cut in cuts:
					dependency_pkl = output_root+file_name+"_treecut"+str(cut)+".pkl"
					if not os.path.exists(dependency_pkl):
						tokens_list=[]
						for tokens_dict in tokens_dict_list:
							tokens_list.append(tokens_dict[cut]) 
						pickle.dump([tokens_list,labels],open(dependency_pkl, 'wb'))
						print('output succees:',dependency_pkl)
						print('shape:',np.array(tokens_list).shape)
					else:
						print("Already exists:",dependency_pkl)

		# entity + tree selection 
		entity_pkl = output_root+file_name+"_entity.pkl"
		if not os.path.exists(entity_pkl):
			tokens_list = CoreNLP.text2tokens_entity(nlp,texts,customized_tokens)
			if dataset in self.opt.pair_set.split(","):
				tokens_list1 = CoreNLP.text2tokens_entity(nlp,texts2,customized_tokens)
				pickle.dump([[tokens_list,tokens_list1],labels],open(entity_pkl, 'wb'))
			else:
				pickle.dump([tokens_list,labels],open(entity_pkl, 'wb'))
			print('output succees:',entity_pkl)
			print('shape:',np.array(tokens_list).shape)
		else:
			print("Already exists:",entity_pkl)
			
		# IDF 
		idf_pkl = output_root+file_name+"_idf.pkl"
		if not os.path.exists(idf_pkl):
			if dataset in self.opt.pair_set.split(","):
				idf_dict = CoreNLP.get_idf_dict(texts2)
				pickle.dump(idf_dict,open(idf_pkl, 'wb'))
			else:
				idf_dict = CoreNLP.get_idf_dict(texts)
				pickle.dump(idf_dict,open(idf_pkl, 'wb'))
			print('output succees:',idf_pkl)
			print('shape:',np.array(idf_dict).shape)
		else:
			print("Already exists:",idf_pkl)

		# IDF + blocks  and IDF+POS+blocks
		# get idf first
		idf_pkl = output_root+"train.csv"+"_idf.pkl"
		self.idf_dict = pickle.load(open(idf_pkl,'rb'))
		# IDF blocks prepare
		for sig_num in [3,4,5,6,7]:
			idf_block_pkl = output_root+file_name+"_idf_block"+str(sig_num)+".pkl"
			if not os.path.exists(idf_block_pkl):
				tokens_list,tokens_list_pos = CoreNLP.text2tokens_blocks_tree(nlp,texts,sig_num,customized_tokens=customized_tokens,idf_dict=self.idf_dict)
				if dataset in self.opt.pair_set.split(","):
					tokens_list1,tokens_list_pos1 = CoreNLP.text2tokens_blocks_tree(nlp,texts2,sig_num,customized_tokens=customized_tokens,idf_dict=self.idf_dict)
					pickle.dump([tokens_list,tokens_list1,tokens_list_pos,tokens_list_pos1,labels],open(idf_block_pkl, 'wb'))
				else:
					pickle.dump([tokens_list,tokens_list_pos,labels],open(idf_block_pkl, 'wb'))
				print('output succees:',idf_block_pkl)
				print('shape:',np.array(idf_block_pkl).shape)
			else:
				print("Already exists:",idf_block_pkl)


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

	# # token selection
	nlp = StanfordCoreNLP(params.corenlp_root)
	# # below is where you need to set your data name
	token_select.token_selection_preparation(nlp = nlp, dataset="WNLI",file_name="train.csv")
	token_select.token_selection_preparation(nlp = nlp, dataset="WNLI",file_name="test.csv")
	nlp.close() # Do not forget to close! The backend server will consume a lot memery.

	# test output some data
	# train,test = token_select.get_train("IMDB",strategy="fulltext",selected_ratio=0.9,cut=1,POS_category="Noun")
	# test_x = test[0]
	# test_y = test[1]
	# for i in range(2):
	# 	print(test_x[i],' -> ',test_y[i],'\n')


