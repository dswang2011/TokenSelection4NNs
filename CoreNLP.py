
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import nltk
from nltk.corpus import stopwords
import re


# POS tag category
noun_list = ['NN','NNS','NNP','NNPS']
verb_list = ['VB', 'VBZ', 'VBD', 'VBG','VBN','VBP']
adjective_list = ['JJ','JJR','JJS']

def get_tokens_POS(nlp,sentence,tag_list):
	token_tag_list = nlp.pos_tag(sentence)
	res_tokens = []
	for k,v in token_tag_list:
		if v in tag_list or k in ['.','!','?',';']:
			res_tokens.append(k)
	return res_tokens
# c(3,1)+c(3,2)+c(3,3) = 7 combinations
def text2token_POS(nlp,text_list):
	Noun=[]
	Verb=[]
	Adjective=[]
	Noun_Verb=[]
	Noun_Adjective=[]
	Verb_Adjective=[]
	Noun_Verb_Adjective=[]

	for text in text_list:
		Noun.append(get_tokens_POS(nlp,text,noun_list))
		Verb.append(get_tokens_POS(nlp,text,verb_list))
		Adjective.append(get_tokens_POS(nlp,text,adjective_list))
		Noun_Verb.append(get_tokens_POS(nlp,text,noun_list+verb_list))
		Noun_Adjective.append(get_tokens_POS(nlp,text,noun_list+adjective_list))
		Verb_Adjective.append(get_tokens_POS(nlp,text,verb_list+adjective_list))
		Noun_Verb_Adjective.append(get_tokens_POS(nlp,text,noun_list+verb_list+adjective_list))
	return [Noun,Verb,Adjective,Noun_Verb,Noun_Adjective,Verb_Adjective,Noun_Verb_Adjective]


# full text
def text2tokens_fulltext(nlp,text_list):
	tokens_list = []
	for text in text_list:
		tokens_list.append(nlp.word_tokenize(text))
	return np.array(tokens_list)

	
# tree Node
class Node(object):
	def __init__(self,data=-1,children=[]):
		self.data=data
		self.children=[]
		self.depth = -1
	def add(self,node):
		self.children.append(node)
	# depth first search
	def iterate_nodes(self):
		queue = []
		queue.append(self)
		while queue:
			v=queue.pop(0)
			print('(',v.data,[node.data for node in v.children],')')
			for child in v.children:
				queue.append(child)
# get depth
def get_depth(node):
	if len(node.children)==0:
		return 0
	temp_max = 0
	for child in node.children:
		child_depth = get_depth(child)
		child.depth = child_depth+1
		if child_depth>temp_max:
			temp_max = child_depth
	# print(node.data,'depth:',node.depth)
	return temp_max+1


# random select 
def random_select(nlp,sentence,select_ratio=1.0):
	word_list = nlp.word_tokenize(sentence)
	selected_count = int(len(word_list)*select_ratio)
	selected_index = np.random.choice(len(word_list),selected_count,replace=False)
	return np.array(word_list)[selected_index].tolist()
def text2tokens_random(nlp,text_list,select_ratio):
	tokens_list = []
	for text in text_list:
		tokens_list.append(random_select(nlp,text,select_ratio))
	return np.array(tokens_list)

# stop word removed selection
def stopword_removed(nlp,sentence):
	stop_words = set(stopwords.words('english'))
	word_list = nlp.word_tokenize(sentence)
	filtered_tokens = [w for w in word_list if not w in stop_words]
	return filtered_tokens
def text2tokens_stopword(nlp,text_list):
	tokens_list = []
	for text in text_list:
		tokens_list.append(stopword_removed(nlp,text))
	return np.array(tokens_list)

# dependency tree based selection 
def build_tree(nlp,sentence):
	tuple_list = nlp.dependency_parse(sentence)
	# print(word_list)
	index2Node = {}
	# buld the tree
	for tag,point,index in tuple_list:
		if index not in index2Node:
			new_node = Node(index)
			index2Node[index] = new_node
		if point not in index2Node:
			new_node = Node(point)
			index2Node[point] = new_node
		point_node = index2Node[point]
		curr_node = index2Node[index]
		point_node.add(curr_node)
	# layer to nodes
	layer2node_list = {}
	layer2node_list[0] = [index2Node[0]]
	try:
		for layer in range(1,10):
			previous_node_list = layer2node_list[layer-1]
			node_list = []
			for node in previous_node_list:
				node_list += node.children
			if len(node_list)==0:
				break
			layer2node_list[layer] = node_list
	except:
		print(sentence,':',tuple_list)
	return layer2node_list

def tree_cut(layer2node_list,nlp,sentence,cut):
	d_thred = len(layer2node_list)-cut
	d_thred = np.max([d_thred,3])	# 3 means 1-2, i.e., two layers
	selected_nodes = []
	for layer in range(1,d_thred):
		if layer>len(layer2node_list)-1:
			break
		selected_nodes += layer2node_list[layer]
	sorted_nodes = sorted(selected_nodes, key=lambda x: x.data)
	word_list = nlp.word_tokenize(sentence)
	return [word_list[node.data-1] for node in sorted_nodes]
# cut mean cutting lower layers, 0 means no cutting, 1 means cut one leaf layer, etc.
def get_token_dependency(nlp,text,cut=0):
	text_tokens=[]
	sentences = re.split('[.!?]',text)
	for sent in sentences:
		if len(sent.strip())<3:
			continue
		if len(sent)>500:
			sent = sent[:450]
		# build tree
		layer2node_list = build_tree(nlp,sent+'.')
		# cut tree
		sent_tokens = tree_cut(layer2node_list,nlp,sent,cut)
		text_tokens+=sent_tokens
		text_tokens+=['.']
	return text_tokens
# text2tokens dependency
def text2tokens_dependency(nlp,text_list,cut=1):
	tokens_list = []
	for text in text_list:
		tokens_list.append(get_token_dependency(nlp,text,cut))
	return np.array(tokens_list)


# cut mean cutting lower layers, 0 means no cutting, 1 means cut one leaf layer, etc.
def get_token_treecuts(nlp,text,cuts=[1,2,3,4,5]):
	text_tokens={}
	sentences = re.split('[.|?|!]',text)
	for sent in sentences:
		if sent.strip()=='':
			continue
		# build tree
		layer2node_list = build_tree(nlp,sent+'.')
		# cut tree
		for cut in cuts:
			if cut not in text_tokens.keys():
				text_tokens[cut]=[]
			sent_tokens = tree_cut(layer2node_list,nlp,sent,cut)
			text_tokens[cut]+=sent_tokens
			text_tokens[cut]+=['.']
	return text_tokens
# text2tokens dependency
def text2tokens_treecuts(nlp,text_list,cuts=[1,2,3,4,5]):
	tokens_dict_list = []
	for text in text_list:
		text_tokens = get_token_treecuts(nlp,text,cuts)
		tokens_dict_list.append(text_tokens)
	return tokens_dict_list

###### test #####
# dependency tree cutting
# res_toknes = get_token_dependency(sentence,4)
# random selection
# print(random_select(sentence,select_ratio=0.2))
# stop word removed
# # print(stopword_removed(sentence))
# texts=["I am dongsheng, and I am from China but live in denmark.","This is test."]
# nlp = StanfordCoreNLP(r'/home/dongsheng/data/resources/stanford-corenlp-full-2018-10-05')
# tokens_list = text2tokens_treecut5(nlp,texts,cuts=[1,2,3])
# print(np.array(tokens_list).shape)
# for tokens in tokens_list:
# 	print('->',tokens)
# nlp.close()
