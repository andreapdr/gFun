# Generalized Funnelling (gFun)

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
  -j, --n_jobs          number of parallel jobs (default is -1, all)
  --nepochs_rnn         number of max epochs to train Recurrent embedder (i.e., -g), default 150.
  --nepochs_bert        number of max epochs to train Bert model (i.e., -g), default 10
  --muse_dir            path to the MUSE polylingual word embeddings (default ../embeddings)
  --gru_wce             deploy WCE embedding as embedding layer of the GRU View Generator
  --gru_dir             set the path to a pretrained GRU model (i.e., -g view generator)
  --bert_dir            set the path to a pretrained mBERT model (i.e., -b view generator)
  --gpus                specifies how many GPUs to use per node
```