### TokenSelection4NNs ###
Do we really need all tokens? Rethinking token selection in neuralnetwork for NLP


### Dependent environment ###
Python3.0, 
libs like: keras, numpy, pickle, etc.


you need to download stanford CoreNLP, the version we used is: "stanford-corenlp-full-2018-10-05" 
You can find in: https://stanfordnlp.github.io/CoreNLP/download.html
after that, put the path in configuration file (refer to configuration section)


### running steps ###
#FIRST: Prepare the data (pre-output the token selection files).
Python File: "token_selection.py".
1. Add your data into path as follows: prepared/your_dataset_name/your_file_name => e.g. prepared/IMBD/train.csv ; prepared/IMBD/test.csv ; etc..
2. Change the parameter in __main__ part, e.g., part of the code is shown below:
-------------------------------------------------------
	nlp = StanfordCoreNLP(params.corenlp_root)
	## below is where you can set your dataset and file_name
	token_select.token_selection_preparation(nlp = nlp, dataset="IMDB",file_name="train.csv")
	token_select.token_selection_preparation(nlp = nlp, dataset="IMDB",file_name="test.csv")
	nlp.close() # Do not forget to close! The backend server will consume a lot memery.

#SECOND: run the neural model.
Python File: "main.py".
1. you need to change the parameters in the function of train_model(), especially for the line below:
-----------------------------
	# strategy can be: fulltext, stopword, random, POS, dependency, 
	train = token_select.get_train(dataset="IMDB",file_name="train.csv",stragety="stopword",POS_category="Noun")

where you need to specify the data and file, and strategy, and then you can run your code. Note if you did not set stragety="POS", the POS_category won't be used.



### configuration ###
Configuration File:"config/config.ini"
1. Basically, you just need to put the standford CoreNLP file path for "corenlp_root"; 
2. And Glove embedding files for "GLOVE_DIR"

Our config.ini looks like below:

	[COMMON]
	MAX_SEQUENCE_LENGTH = 150
	MAX_SEQUENCE_LENGTH_contatenate = 150 
	MAX_NB_WORDS = 20000   
	EMBEDDING_DIM = 100
	VALIDATION_SPLIT = 0.1
	batch_size = 64
	epoch_num = 100
	dropout_rate = 0.2
	hidden_unit_num = 100
	hidden_unit_num_second = 100
	cell_type = gru
	contatenate = 1
	lr= 0.001
	corenlp_root=/home/dongsheng/data/resources/stanford-corenlp-full-2018-10-05
	GLOVE_DIR = /home/dongsheng/data/resources/glove
	dataset_dir = input/dataset
	model= bilstm2
