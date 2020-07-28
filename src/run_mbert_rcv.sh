#!/usr/bin/env bash

dataset_path=/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run
logfile=../log/log_Mbert_rcv.csv

runs='1 2 3 4 5 6 7 8 9'
for run in $runs
do
  dataset=$dataset_path$run.pickle
  python new_mbert.py --dataset $dataset --log-file $logfile --nepochs=5 --weight_decay=0.01 --lr=1e-5
done
