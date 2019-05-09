# TokenSelection4NNs
Do we really need all tokens? Rethinking token selection in neuralnetwork for NLP


# Dependent environment
Python3.0

you need to download stanford CoreNLP, the version we used is: "stanford-corenlp-full-2018-10-05" 
You can find in: https://stanfordnlp.github.io/CoreNLP/download.html


# running steps:
1. Prepare the data (token selection output)
	"token_selection.py"
change the data path in the part of __main__ (before that add the data file of yours) 

2. run the model
	"main.py"
First, you need to change the parameters in the function of train_model(), especially for the line below:
--------------
train = token_select.get_train(dataset="IMDB",file_name="train.csv",stragety="stopword",POS_category="Noun")
----------------
where you need to specify the data and file, and strategy, and then you can run your code. Note if you did not set stragety="POS", the POS_category won't be used.




