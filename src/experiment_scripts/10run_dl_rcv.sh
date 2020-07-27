#!/usr/bin/env bash

dataset_path=/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run
logfile=../log/log10run_dl_rcv.csv

runs='0 1 2 3 4 5 6 7 8 9'
for run in $runs
do
  dataset=$dataset_path$run.pickle
  python main_deep_learning.py $dataset --log-file $logfile --pretrained --supervised --posteriors --tunable --plotmode --test-each 20
done
