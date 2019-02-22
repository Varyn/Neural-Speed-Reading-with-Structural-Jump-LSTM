The code is written in python 3.5.4, using tenforflow 1.7.
We use GloVe word embedding from http://nlp.stanford.edu/data/glove.840B.300d.zip loaded through gensim in the code. This should be placed in the data directory.

network_run.py is the run file, and contains the configuration to run on the dpbedia dataset, which lies in the data folder.
network_argent.py is the implementation of the agents running the network, and the reward function.
network.py contains the code for the network.
data_generator.py contain the code for handling the input.
preprocessing/datasetMaker.py generated the input files expected by network_run.py

To run as done in our paper, there are 2 steps:
1) Full read: set partial_read_until equal=30000, which runs a full read model and saves the best model based on the validation data.
2) Speed read: set partial_read_until=1 and use_pretrained=True after having placed a saved full read model in pretrained/dbpedia/w_entropy, where the full read model can be found in the results/<run name>/fullread directory after step (1).

To allow the code to be tested on GPUs with little memory we have attached files for "dbpedia_medium" in the data directory, which is a medium sized version of DBPedia - this was not used on our experimental evaluation.