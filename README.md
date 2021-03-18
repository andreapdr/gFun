# Generalized Funnelling (Heterogeneous Document Embeddings) Code
This repository contains the Python code developed for the experiments conducted pertaining Heterogeneous Document Embeddings in both traditional machine learning and deep learning sceneario (Msc Thesis).
Concerning traditional machine learning the code implements variants to Funnelling algoirthm (TAT) proposed in the article "Esuli, A., Moreo, A., & Sebastiani, F. (2019). [Funnelling: A New Ensemble Method for Heterogeneous Transfer Learning and Its Application to Cross-Lingual Text Classification](https://dl.acm.org/citation.cfm?id=3326065). ACM Transactions on Information Systems (TOIS), 37(3), 37.".

To form document representations we deployed publicly available word-embeddings:
* MUSE  "A. Conneau, G. Lample, L. Denoyer, MA. Ranzato, H. Jégou, (2018). [Word Translation without Parallel Data](https://arxiv.org/pdf/1710.04087.pdf)"
As well as a method to build supervised word-embeddings:
* Word-Class Embeddings "Moreo, A, Esuli, A. & Sebastiani, F. (2019). [Word-Class Embeddings for Multiclass Text Classification](https://arxiv.org/abs/1911.11506.pdf)"


This code has been used to produce all experimental results reported.

## Datasets

The datasets we used to run our experiments include:
* RCV1/RCV2: a _comparable_ corpus of Reuters newstories
* JRC-Acquis: a _parallel_ corpus of legislative texts of the European Union

The datasets need to be built before running any experiment.
This process requires _downloading_, _parsing_, _preprocessing_, _splitting_, and _vectorizing_.
The datasets we generated and used in our experiments can be directly downloaded (in vector form) from [here](http://hlt.isti.cnr.it/funnelling/).

## Reproducing the Experiments

Most of the experiments were run using either the script [main_deep_learning.py] and [main_multimodal_cls.py].
These scripts can be run with different command line arguments to reproduce all experimental settings.

Run it with _-h_ or _--help_ to show this help.

## Usage
```commandline
usage: main.py [-h] [-o CSV_DIR] [-x] [-w] [-m] [-b] [-g] [-c] [-n NEPOCHS]
               [-j N_JOBS] [--muse_dir MUSE_DIR] [--gru_wce]
               [--gru_dir GRU_DIR] [--bert_dir BERT_DIR] [--gpus GPUS]
               dataset

Run generalized funnelling, A. Moreo, A. Pedrotti and F. Sebastiani (2020).

positional arguments:
  dataset               Path to the dataset

optional arguments:
  -h, --help            show this help message and exit
  -o, --output          result file (default ../csv_logs/gfun/gfun_results.csv)
  -x, --post_embedder   deploy posterior probabilities embedder to compute document embeddings
  -w, --wce_embedder    deploy (supervised) Word-Class embedder to the compute document embeddings
  -m, --muse_embedder   deploy (pretrained) MUSE embedder to compute document embeddings
  -b, --bert_embedder   deploy multilingual Bert to compute document embeddings
  -g, --gru_embedder    deploy a GRU in order to compute document embeddings
  -c, --c_optimize      optimize SVMs C hyperparameter
  -j, --n_jobs          number of parallel jobs, default is -1 i.e., all 
  --nepochs_rnn         number of max epochs to train Recurrent embedder (i.e., -g), default 150
  --nepochs_bert        number of max epochs to train Bert model (i.e., -g), default 10
  --patience_rnn        set early stop patience for the RecurrentGen, default 25
  --patience_bert       set early stop patience for the BertGen, default 5
  --batch_rnn           set batchsize for the RecurrentGen, default 64
  --batch_bert          set batchsize for the BertGen, default 4
  --muse_dir            path to the MUSE polylingual word embeddings (default ../embeddings)
  --gru_wce             deploy WCE embedding as embedding layer of the GRU View Generator
  --rnn_dir             set the path to a pretrained RNN model (i.e., -g view generator)
  --bert_dir            set the path to a pretrained mBERT model (i.e., -b view generator)
  --gpus                specifies how many GPUs to use per node
```

## Requirements
```commandline
transformers==2.11.0
pandas==0.25.3
numpy==1.17.4
joblib==0.14.0
tqdm==4.50.2
pytorch_lightning==1.1.2
torch==1.3.1
nltk==3.4.5
scipy==1.3.3
rdflib==4.2.2
torchtext==0.4.0
scikit_learn==0.24.1
```
