

# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np

# corpus = [
# 	'This is the first document.',
# 	'This document is the second document.',
# 	'And this is the third one.',
# 	'Is this the first document?',
# 	'why there is no stop word',
# 	'I do not know why. very weird'
# 	]

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)
# idf = vectorizer.idf_
# print('idf:',idf)
# print('dict_idf:',dict(zip(vectorizer.get_feature_names(), idf)))
# print('stop:',vectorizer.stop_words_)

# from sklearn.feature_extraction.text import TfidfVectorizer
# corpus = ["This is very strange",
#           "This is very nice"]
# vectorizer = TfidfVectorizer(
#                         use_idf=True, # utiliza o idf como peso, fazendo tf*idf
#                         norm=None, # normaliza os vetores
#                         smooth_idf=False, #soma 1 ao N e ao ni => idf = ln(N+1 / ni+1)
#                         sublinear_tf=False, #tf = 1+ln(tf)
#                         binary=False,
#                         min_df=1, max_df=1.0, max_features=None,
#                         strip_accents='unicode', # retira os acentos
#                         ngram_range=(1,1), preprocessor=None,              stop_words=None, tokenizer=None, vocabulary=None
#              )
# X = vectorizer.fit_transform(corpus)
# idf = vectorizer.idf_
# print(dict(zip(vectorizer.get_feature_names(), idf)))
# print('stop:',vectorizer.stop_words_)

import spacy

nlp = spacy.load("en")

doc = nlp("We try to explicitly describe the geometry of the edges of the images.")

for np in doc.noun_chunks: # use np instead of np.text
    print(np)

print()

# code to recursively combine nouns
# 'We' is actually a pronoun but included in your question
# hence the token.pos_ == "PRON" part in the last if statement
# suggest you extract PRON separately like the noun-chunks above

index = 0
nounIndices = []
for token in doc:
    # print(token.text, token.pos_, token.dep_, token.head.text)
    if token.pos_ == 'NOUN':
        nounIndices.append(index)
    index = index + 1


print(nounIndices)
for idxValue in nounIndices:
    doc = nlp("We try to explicitly describe the geometry of the edges of the images.")
    span = doc[doc[idxValue].left_edge.i : doc[idxValue].right_edge.i+1]
    span.merge()

    for token in doc:
        if token.dep_ == 'dobj' or token.dep_ == 'pobj' or token.pos_ == "PRON":
            print(token.text)
            