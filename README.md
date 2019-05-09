### TokenSelection4NNs ###
Do we really need all tokens? Rethinking token selection in neuralnetwork for NLP


### Dependent environment ###
Python3.0

you need to download stanford CoreNLP, the version we used is: "stanford-corenlp-full-2018-10-05" 
You can find in: https://stanfordnlp.github.io/CoreNLP/download.html
after that, put the path in configuration file (refer to configuration section)


### running steps ###
1. Prepare the data (pre-output the token selection files)
File: "token_selection.py"
1.1. Add your data as follows: prepared/your_dataset_name/your_file_name => e.g. prepared/IMBD/train.csv   prepared/IMBD/test.csv
2.2. Change the parameter in __main__ part, e.g., part of the data is shown below:
-------------------------------------------------------------
	nlp = StanfordCoreNLP(params.corenlp_root)
    # below is where you can set your dataset and file_name
    token_select.token_selection_preparation(nlp = nlp, dataset="IMDB",file_name="train.csv")
    token_select.token_selection_preparation(nlp = nlp, dataset="IMDB",file_name="test.csv")
    nlp.close() # Do not forget to close! The backend server will consume a lot memery.
-------------------------------------------------------------

2. run the model
	"main.py"
First, you need to change the parameters in the function of train_model(), especially for the line below:

train = token_select.get_train(dataset="IMDB",file_name="train.csv",stragety="stopword",POS_category="Noun")

where you need to specify the data and file, and strategy, and then you can run your code. Note if you did not set stragety="POS", the POS_category won't be used.



### configuration ###
File:"config/config.ini"
1. Basically, you just need to put the standford CoreNLP file path for "corenlp_root"; 
2. Glove embedding in "GLOVE_DIR"

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
