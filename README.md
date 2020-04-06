# Heterogeneous Document Embeddings Code
This repository contains the Python code developed for the experiments conducted pertaining Heterogeneous Document Embeddings in both traditional machine learning and deep learning sceneario (Msc Thesis).
Concerning traditional machine learning the code implements variants to Funnelling algoirthm (TAT) proposed in the article "Esuli, A., Moreo, A., & Sebastiani, F. (2019). [Funnelling: A New Ensemble Method for Heterogeneous Transfer Learning and Its Application to Cross-Lingual Text Classification](https://dl.acm.org/citation.cfm?id=3326065). ACM Transactions on Information Systems (TOIS), 37(3), 37.".

To form document representations we deployed publicly available word-embeddings:
* MUSE  A. Conneau, G. Lample, L. Denoyer, MA. Ranzato, H. Jégou, (2018). [Word Translation without Parallel Data](https://arxiv.org/pdf/1710.04087.pdf)"
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

```
Usage: main_multimodal_cls.py [options]

Options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output=OUTPUT
                        Result file
  -P, --posteriors      Add posterior probabilities to the document embedding
                        representation
  -S, --supervised      Add supervised (Word-Class Embeddings) to the document
                        embedding representation
  -U, --pretrained      Add pretrained MUSE embeddings to the document
                        embedding representation
  --l2                  Activates l2 normalization as a post-processing for
                        the document embedding views
  --allprob             All views are generated as posterior probabilities.
                        This affects the supervised and pretrained embeddings,
                        for which a calibrated classifier is generated, which
                        generates the posteriors
  --feat-weight=FEAT_WEIGHT
                        Term weighting function to weight the averaged
                        embeddings
  -w WE_PATH, --we-path=WE_PATH
                        Path to the MUSE polylingual word embeddings
  -s SET_C, --set_c=SET_C
                        Set the C parameter
  -c, --optimc          Optimize hyperparameters
  -j N_JOBS, --n_jobs=N_JOBS
                        Number of parallel jobs (default is -1, all)
  -p MAX_LABELS_S, --pca=MAX_LABELS_S
                        If smaller than number of target classes, PCA will be
                        applied to supervised matrix.
  -r, --remove-pc       Remove common component when computing dot product of
                        word embedding matrices
  -z, --zscore          Z-score normalize matrices (WCE and MUSE)
  -a, --agg             Set aggregation function of the common Z-space to
                        average (Default: concatenation)

```

```
Usage: main_deep_learning.py [options]

Options:
 positional arguments:
  datasetpath           path to the pickled dataset

optional arguments:
  -h, --help            show this help message and exit
  --batch-size int      input batch size (default: 100)
  --batch-size-test int
                        batch size for testing (default: 250)
  --nepochs int         number of epochs (default: 200)
  --patience int        patience for early-stop (default: 10)
  --plotmode            in plot mode executes a long run in order to generate
                        enough data to produce trend plots (test-each should
                        be >0. This mode is used to produce plots, and does
                        not perform an evaluation on the test set.
  --hidden int          hidden lstm size (default: 512)
  --lr float            learning rate (default: 1e-3)
  --weight_decay float  weight decay (default: 0)
  --sup-drop [0.0, 1.0]
                        dropout probability for the supervised matrix
                        (default: 0.5)
  --seed int            random seed (default: 1)
  --svm-max-docs int    maximum number of documents by language used to train
                        the calibrated SVMs (only used if --posteriors is
                        active)
  --log-interval int    how many batches to wait before printing training
                        status
  --log-file str        path to the log csv file
  --test-each int       how many epochs to wait before invoking test (default:
                        0, only at the end)
  --checkpoint-dir str  path to the directory containing checkpoints
  --net str             net, one in {'rnn'}
  --pretrained          use MUSE pretrained embeddings
  --supervised          use supervised embeddings
  --posteriors          concatenate posterior probabilities to doc embeddings
  --learnable int       dimension of the learnable embeddings (default 0)
  --val-epochs int      number of training epochs to perform on the validation
                        set once training is over (default 1)
  --we-path str         path to MUSE pretrained embeddings
  --max-label-space int
                        larger dimension allowed for the feature-label
                        embedding (if larger, then PCA with this number of
                        components is applied (default 300)
  --force               do not check if this experiment has already been run
  --tunable             pretrained embeddings are tunable from the begining
                        (default False, i.e., static)

```

