```
usage: main.py [-h] [-o CSV_DIR] [-x] [-w] [-m] [-b] [-g] [-c] [-n NEPOCHS]
               [-j N_JOBS] [--muse_dir MUSE_DIR] [--gru_wce]
               [--gru_dir GRU_DIR] [--bert_dir BERT_DIR] [--gpus GPUS]
               dataset

Run generalized funnelling, A. Moreo, A. Pedrotti and F. Sebastiani (2020).

positional arguments:
  dataset               Path to the dataset

optional arguments:
  -h, --help            show this help message and exit
  -o CSV_DIR, --output CSV_DIR
                        Result file (default ../csv_logs/gfun/gfun_results.csv)
  -x, --post_embedder   deploy posterior probabilities embedder to compute
                        document embeddings
  -w, --wce_embedder    deploy (supervised) Word-Class embedder to the compute
                        document embeddings
  -m, --muse_embedder   deploy (pretrained) MUSE embedder to compute document
                        embeddings
  -b, --bert_embedder   deploy multilingual Bert to compute document
                        embeddings
  -g, --gru_embedder    deploy a GRU in order to compute document embeddings
  -c, --c_optimize      Optimize SVMs C hyperparameter
  -n NEPOCHS, --nepochs NEPOCHS
                        Number of max epochs to train Recurrent embedder
                        (i.e., -g)
  -j N_JOBS, --n_jobs N_JOBS
                        Number of parallel jobs (default is -1, all)
  --muse_dir MUSE_DIR   Path to the MUSE polylingual word embeddings (default
                        ../embeddings)
  --gru_wce             Deploy WCE embedding as embedding layer of the GRU
                        View Generator
  --gru_dir GRU_DIR     Set the path to a pretrained GRU model (i.e., -g view
                        generator)
  --bert_dir BERT_DIR   Set the path to a pretrained mBERT model (i.e., -b
                        view generator)
  --gpus GPUS           specifies how many GPUs to use per node
```