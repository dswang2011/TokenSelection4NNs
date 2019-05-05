# Word index=1;text=Barack;lemma=Barack;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=4;dependency_relation=nsubj:pass


# global indicator -> root oriented connection across sentences; 
# 0 is empty word
import numpy as np
import pprint

def get_root(sent):
	sub_index = 0
	root_index = 1
	# find the root first
	for word in sent.words:
		if word.dependency_relation.strip()=='root':
			root_index = word.index
	# find the subject for the root
	for word in sent.words:
		if word.governor==root_index and word.dependency_relation.strip() =='nsubj:pass':
			sub_index = word.index
		print(word.index,':',word.text,'-> govener:',word.governor,':',word.dependency_relation)
	return sub_index,root_index

# localized: pronoun 
def get_pronoun_oriented(sent):
	pronoun_index = 0
	pred_index = 0
	# find the pronoun
	for word in sent.words:
		if word.text.strip() in ['pppc','pppcs']:
			pronoun_index=word.index
			if word.text.strip() == 'pppcs':
				ps_governor = word.governor
				pred_index = sent.words[word.governor].governor
			else:
				pred_index = word.governor
	return pronoun_index,pred_index

# localized: names
def get_a_oriented(sent):
	a_index = 0
	pred_index = 0
	# find the pronoun
	for word in sent.words:
		if word.text.strip() in ['aaac']:
			a_index = word.index
			pred_index = word.governor
	return a_index,pred_index	
	# find the predicate for the pronoun

# localized: names
def get_b_oriented(sent):
	b_index = 0
	pred_index = 0
	# find the pronoun
	for word in sent.words:
		if word.text.strip() in ['bbbc']:
			b_index = word.index
			pred_index = word.governor
	return b_index,pred_index	
	# find the predicate for the pronoun


# def get_global_encoding(doc):
def get_global_predicate(doc):
	doc_global_output = []
	for sent in doc.sentences:
		for word in sent.words:
			if word.dependency_relation.rstrip() == "root":
				head = int(word.index)-1
		doc_global_output += [{'head':head, 'body':[]}]
	return doc_global_output


def get_text_matrix(doc):
	matrix = []
	for sent in doc.sentences:
		sentece = []
		for word in sent.words:
			sentece += [word.text]
		matrix += [sentece]
	return matrix


def get_mention_predicate(doc):
	dic_doc_mention ={}
	i=0
	for sent in doc.sentences:
		for word in sent.words:
			if word.text == "PPPCS":
				# print('govern:',word.governor)
				PPPCS_predicate = int(sent.words[word.governor-1].governor)-1
				dic_doc_mention['pppcs'] = {"predicate": [i, PPPCS_predicate], "mention": [i, int(word.index)-1]}
			elif word.text == "PPPC":
				PPPC_predicate = int(word.governor)-1
				dic_doc_mention['pppc'] = {"predicate": [i, PPPC_predicate], "mention": [i, int(word.index)-1]}
			elif word.text == "AAAC":
				AAAC_predicate = int(word.governor)-1
				dic_doc_mention['aaac'] = {"predicate": [i, AAAC_predicate], "mention": [i, int(word.index)-1]}
			elif word.text == "BBBC":
				BBBC_predicate = int(word.governor)-1
				dic_doc_mention['bbbc'] = {"predicate": [i, BBBC_predicate], "mention": [i, int(word.index)-1]}
		i += 1
	return dic_doc_mention




