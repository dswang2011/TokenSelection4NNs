
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import nltk
from nltk.corpus import stopwords

nlp = StanfordCoreNLP(r'/home/dongsheng/data/resources/stanford-corenlp-full-2018-10-05')

# sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
sentence = 'This is a sample sentence, showing off the stop words filtration.'
# print('Tokenize:', nlp.word_tokenize(sentence))
# print('Part of Speech:', nlp.pos_tag(sentence))
print('Named Entities:', nlp.ner(sentence))
# print('Constituency Parsing:', nlp.parse(sentence))
# print('Dependency Parsing:', nlp.dependency_parse(sentence))




# punctuation tags: . :
noun_list = ['NN','NNS','NNP','NNPS']
# VB, VBZ, VBD, VBG, VBN, VBP
verb_list = ['VB', 'VBZ', 'VBD', 'VBG','VBN','VBP']
adjective_list = ['JJ','JJR','JJS']

def get_tokens_POS(sentence,tag_list):
	token_tag_list = nlp.pos_tag(sentence)
	res_tokens = []
	for k,v in token_tag_list:
		if v in tag_list:
			res_tokens.append(k)
	return res_tokens
# c(3,1)+c(3,2)+c(3,3) = 7 combinations
def text2token_POS(text_list,label_list):
	Noun=[]
	Verb=[]
	Adjective=[]
	Noun_Verb=[]
	Noun_Adjective=[]
	Verb_Adjective=[]
	Noun_Verb_Adjective=[]

	for text in text_list:
		Noun.append(get_tokens_POS(text,noun_list))
		Verb.append(get_tokens_POS(text,verb_list))
		Adjective.append(get_tokens_POS(text,adjective_list))
		Noun_Verb.append(get_tokens_POS(text,noun_list+verb_list))
		Noun_Adjective.append(get_tokens_POS(text,noun_list+adjective_list))
		Verb_Adjective.append(get_tokens_POS(text,verb_list+adjective_list))
		Noun_Verb_Adjective.append(get_tokens_POS(text,noun_list+verb_list+adjective_list))
	return [Noun,Verb,Adjective,Noun_Verb,Noun_Adjective,Verb_Adjective,Noun_Verb_Adjective,label_list]


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
def random_select(sentence,select_ratio=1.0):
	word_list = nlp.word_tokenize(sentence)
	selected_count = int(len(word_list)*select_ratio)
	selected_index = np.random.choice(len(word_list),selected_count,replace=False)
	return np.array(word_list)[selected_index].tolist()
def text2tokens_random(text_list,label_list,select_ratio):
	sequence = []
	for text in text_list:
		sequence.append(random_select(text,select_ratio))
	return [sequence,label_list]


def stopword_removed(sentence):
	stop_words = set(stopwords.words('english'))
	word_list = nlp.word_tokenize(sentence)
	filtered_tokens = [w for w in word_list if not w in stop_words]
	return filtered_tokens
def text2tokens_stopword(text_list,label_list):
	sequence = []
	for text in text_list:
		sequence.append(stopword_removed(text))
	return [sequence,label_list]


# cut mean cutting lower layers, 0 means no cutting, 1 means cut one leaf layer, etc.
def get_token_dependency(sentence,cut=0):
	tuple_list = nlp.dependency_parse(sentence)
	# print(tuple_list)
	word_list = nlp.word_tokenize(sentence)
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
	queue = []
	layer2node_list = {}
	queue.append(index2Node[0])
	layer2node_list[0] = [index2Node[0]]
	for layer in range(1,10):
		previous_node_list = layer2node_list[layer-1]
		node_list = []
		for node in previous_node_list:
			node_list += node.children
		if len(node_list)==0:
			break
		layer2node_list[layer] = node_list

	# get the first (tree_dept-cut) layer nodes
	d_thred = len(layer2node_list)-cut
	d_thred = np.max([d_thred,3])	# 3 means 1-2, i.e., two layers
	print('d_thred',d_thred)
	selected_nodes = []
	for layer in range(1,d_thred):
		if layer>len(layer2node_list)-1:
			break
		selected_nodes += layer2node_list[layer]
	sorted_nodes = sorted(selected_nodes, key=lambda x: x.data)
	print([word_list[node.data-1] for node in sorted_nodes])
	return [word_list[node.data-1] for node in sorted_nodes]

def text2tokens_dependency(text_list,label_list,cut=1):
	sequence = []
	for text in text_list:
		sequence.append(get_token_dependency(text,cut))
	return [sequence,label_list]

###### test #####
# dependency tree cutting
# res_toknes = get_token_dependency(sentence,4)
# random selection
# print(random_select(sentence,select_ratio=0.2))
# stop word removed
# print(stopword_removed(sentence))

nlp.close() # Do not forget to close! The backend server will consume a lot memery.