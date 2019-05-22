
import os
#task: 1. get tokenized diction; 2. 

#import stanfordnlp
import NLP
import numpy as np
import codecs

# global tool
nlp = None



punctuation_list = [',',':',';','.','!','?','...','…','。']
# punctuation_list = ['.']


def get_embedding_dict(GLOVE_DIR):
	embeddings_index = {}
	f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding="utf-8")
	for line in f:
		if line.strip()=='':
			continue
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	# customized dict
	f =  codecs.open(os.path.join(GLOVE_DIR, 'customized.100d.txt'),encoding="utf-8")  #
	for line in f:
		if line.strip()=='':
			continue
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	return embeddings_index


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()


def tokenizer(tokens_list,MAX_NB_WORDS):
	index = 1
	word_index = {}
	for tokens in tokens_list:
		for token in tokens:
			# add to word_index
			if len(word_index)<MAX_NB_WORDS:
				token=token.lower()
				if token in word_index.keys():
					continue
				else:
					word_index[token] = index
					index+=1
	return word_index

# input is the generalized text; 
def text_to_sequences(gene_texts,word_index, MAX_SEQUENCE_LENGTH):
	sequences = []
	if nlp is None:
		nlp=stanfordnlp.Pipeline(use_gpu=False)
	# nlp = stanfordnlp.Pipeline()
	for text in gene_texts:
		doc = nlp(text)
		txt_matrix = NLP.get_text_matrix(doc)	# doc matrix (array)
		mention_pred = NLP.get_mention_predicate(doc)	# local
		global_pred = NLP.get_global_predicate(doc)	# global
		sequence = []
		
		# == process this text matrix
		for i in range(len(txt_matrix)):
			sent_arr = txt_matrix[i]
			for j in range(len(sent_arr)):
				token = sent_arr[j].lower()
				token_index = 0	
				if token in word_index.keys():
					token_index = word_index[token]
				# local encoding (predicates of mentions)
				local_encoding = 0 
				if token in ['aaac','bbbc','pppc','pppcs']:
					pred_pos = mention_pred[token]['predicate']
					pred_token = txt_matrix[pred_pos[0]][pred_pos[1]]
					if pred_token in word_index.keys():
						local_encoding = word_index[pred_token]
				# global encoding (punctuations or predicates)
				global_encoding = 0
				if token in punctuation_list:
					global_encoding = token_index
				else:
					if global_pred[i]['head']==j:
						global_encoding = token_index
				# concatenate
				concate = [token_index,local_encoding,global_encoding]
				sequence+=concate # add to the list
		if len(sequence)>MAX_SEQUENCE_LENGTH:
			sequence = sequence[:MAX_SEQUENCE_LENGTH]
		else:
			sequence = np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()+sequence
		sequences.append(sequence)
	return np.asarray(sequences,dtype=int)


# input is the generalized text; 
def tokens_list_to_sequences(tokens_lists,word_index, MAX_SEQUENCE_LENGTH):
	sequences = []
	for tokens in tokens_lists:
		sequence = []
		for token in tokens:
			token = token.lower()
			if token in word_index.keys():
				token_index = word_index[token]
				sequence.append(token_index)
		if len(sequence)>MAX_SEQUENCE_LENGTH:
			sequence = sequence[:MAX_SEQUENCE_LENGTH]
		else:
			sequence = np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()+sequence
		# print('seq:',sequence)
		sequences.append(sequence)
	return np.asarray(sequences,dtype=int)

# input is the generalized text; 
 

def docs_to_sequences_suffix(docs,word_index, MAX_SEQUENCE_LENGTH, contatenate=0):

	sequences = []
	a = 1
	for doc in docs:
		# print("Doc in docs:", a)
		a+=1
		txt_matrix = NLP.get_text_matrix(doc)	# doc matrix (array)
		# txt_matrix = np.asarray(txt_matrix)
		mention_pred = NLP.get_mention_predicate(doc)	# local
		global_pred = NLP.get_global_predicate(doc)	# global
		sequence = []

		# == process this text matrix
		attentions = []
		attentions+=[word_index['.']]
		for i in range(len(txt_matrix)):
			sent_arr = txt_matrix[i]
			if i==0:
				sequence+=[word_index['.']]
			for j in range(len(sent_arr)):
				token = sent_arr[j].lower()
				token_index = 0
				if token in word_index.keys():
					token_index = word_index[token]
				if contatenate==1:

					sequence += [token_index]
				# local encoding
				pred_index = 0
				if token in ['aaac','bbbc','pppc','pppcs']:
					possessive = 0
					if len(sent_arr)>(j+1) and sent_arr[j+1].lower()=="'s":
						possessive = 1
					pred_pos = mention_pred[token]['predicate']
					pred_token = txt_matrix[pred_pos[0]][pred_pos[1]]
					if pred_token in word_index.keys():
						pred_index = word_index[pred_token]
					if pred_pos[1]>=j: # predicate occur after mention
						attentions+=[token_index,pred_index]
					else:
						attentions+=[pred_index,token_index]
				# global encoding
				if token in punctuation_list:
					attentions+= [token_index]
				else:
					if global_pred[i]['head']==j:
						attentions+=[token_index]
		# contatenate 
		sequence += attentions
		# print('seq/att:',len(sequence),len(attentions))
		# padding
		if len(sequence)>MAX_SEQUENCE_LENGTH:
			sequence = sequence[:MAX_SEQUENCE_LENGTH]
		else:
			sequence = np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()+sequence
		# print('seq:',sequence)
		sequences.append(sequence)
	return np.asarray(sequences,dtype=int)

# input is the generalized text; 
def text_to_sequences_suffix(gene_texts,word_index, MAX_SEQUENCE_LENGTH):
	sequences = []
	for text in gene_texts:
		doc = nlp(text)
		txt_matrix = NLP.get_text_matrix(doc)	# doc matrix (array)
		mention_pred = NLP.get_mention_predicate(doc)	# local
		global_pred = NLP.get_global_predicate(doc)	# global
		sequence = []
		
		# == process this text matrix
		attentions = []
		for i in range(len(txt_matrix)):
			sent_arr = txt_matrix[i]	
			for j in range(len(sent_arr)):
				token = sent_arr[j].lower()
				token_index = 0	
				if token in word_index.keys():
					token_index = word_index[token]
				# local encoding
				pred_index = 0 
				if token in ['aaac','bbbc','pppc','pppcs']:
					pred_pos = mention_pred[token]['predicate']
					pred_token = txt_matrix[pred_pos[0]][pred_pos[1]]
					if pred_token in word_index.keys():
						pred_index = word_index[pred_token]
					if pred_pos[1]>=j: # predicate occur after mention
						attentions+=[token_index,pred_index]
					else:
						attentions+=[pred_index,token_index]
				# global encoding
				if token in punctuation_list:
					attentions+= [token_index]
				else:
					if global_pred[i]['head']==j:
						attentions+=[token_index]
		# contatenate
		sequence += attentions
		# padding
		if len(sequence)>MAX_SEQUENCE_LENGTH:
			sequence = sequence[:MAX_SEQUENCE_LENGTH]
		else:
			sequence = np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()+sequence
		# print('seq:',sequence)
		sequences.append(sequence)
	return np.asarray(sequences,dtype=int)


def load_data(tsv_file_path,mode= "train"):
    with open(tsv_file_path, encoding='utf8') as f:
        content = f.readlines()
    content = [x.rstrip() for x in content]
    header = content[0]
    res = []
    for line in content[1:]:
        data = DatasetSchema(line)
        orig_txt = data.get_text()
        generalized_txt = data.get_generalized_text()
        # below is to get exact sentences
        sentences = generalized_txt.split('.')
        exact_sents = []
        for sent in sentences:
            if 'AAAC' in sent or 'BBBC' in sent or 'PPPC' in sent or 'PPPCS' in sent:
                exact_sents.append(sent)
        exact_txt = '.'.join(exact_sents)
        # end of previous below

        if mode == "train":
            label_A = data.get_A_coref()
            label_B = data.get_B_coref()
            if label_A in ['TRUE','True','true'] and label_B in ['FALSE','False','false']:
                label = 0
            elif label_B in ['TRUE','True','true'] and label_A in ['FALSE','False','false']:
                label = 1
            else:
                label = 2
            res.append([orig_txt,exact_txt,label])
        else:
            samp_id = data.get_id()
            res.append([orig_txt,exact_txt,samp_id])
    return np.array(res)

import csv
#### uncomment this to use for the IMDB dataset ##########
# def load_classification_data(file_path,hasHead=0):
# 	texts=[]
# 	labels=[]
# 	with open(file_path, encoding='utf8') as f:
# 		csv_reader = csv.reader(f, delimiter='\t')
# 		for row in csv_reader:
# 			texts.append(row[0].strip())
# 			label = '0'
# 			for i in range(1,len(row)):
# 				if row[i].strip() in ['0','1']:
# 					label = row[i].strip()
# 			labels.append(label)
# 	# print('labels:',labels)
#
# 	return [texts,labels]

#### THIS IS TO RUN FOR GAP ######
def load_classification_data(file_path,hasHead=0):
	texts=[]
	labels=[]
	with open(file_path, encoding='utf8') as f:
		csv_reader = csv.reader(f, delimiter='\t')
		for row in csv_reader:
			texts.append(row[0].strip())
			# label = '0'
			for i in range(1,len(row)):
			# 	if row[i].strip() in ['0','1']:
				label = row[i].strip()
				# print(label)
			labels.append(label)
	# print('labels:',labels)

	return [texts,labels]


def load_pair_data(file_path,hasHead=0):
	texts1,texts2=[],[]
	labels=[]
	with open(file_path, encoding='utf8') as f:
		csv_reader = csv.reader(f, delimiter='\t')
		for row in csv_reader:
			max_snippets = np.maximum(5,len(row)-3)
			texts1.append(row[1].strip())
            text2 = ' '.join(rows[3:max_snippets])
            texts2.append(text2.strip())
			labels.append(row[2].strip())
	return [[texts1,texts2],labels]

def load_triple_data(file_path):
	triples=[]
	labels=[]
	with open(file_path,'r',encoding='utf8') as f:
		for line in f:
			strs = line.split('\t')
			triples.append(strs[0].strip())
			label = '0'
			for i in range(1,len(strs)):
				if strs[i].strip() in ['0','1']:
					label = strs[i].strip()
			labels.append(label)
	return [triples,labels]


def get_texts_from_folder(directory):
	texts = []
	for filename in os.listdir(directory):
		if filename.endswith(".txt"):
			file_path = os.path.join(directory, filename)
			file = open(file_path,'r')
			lines = file.readlines()
			texts.append(' '.join(lines).replace('\n',''))
	return texts
# load MR
from sklearn.utils import shuffle
def load_mr_data(folder):
	pos_texts = get_texts_from_folder(folder+'/'+'pos/')
	neg_texts = get_texts_from_folder(folder+'/'+'neg/')
	pos_labels = np.ones(len(pos_texts),dtype=int)
	neg_labels = np.zeros(len(neg_texts),dtype=int)
	texts = pos_texts+neg_texts
	labels = pos_labels.tolist()+neg_labels.tolist()
	X,y = shuffle(texts, labels, random_state=0)
	return [X,y]
# Process MR 
# def write_content(file_path,content):
# 	with open(file_path,'a',encoding='utf8') as fw:
# 		fw.write(content)
# X,y = load_mr_data("/home/dongsheng/code/TokenSelection4NNs/prepared/MR/txt_sentoken/")
# for i in range(len(X)):
# 	content = X[i].replace('\t',' ').strip()+'\t'+str(y[i])
# 	write_content("mr.csv",content+'\n')
