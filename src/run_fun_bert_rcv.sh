#!/usr/bin/env bash

#dataset_path=/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run
#logfile=../log/log_FunBert_rcv_static.csv
#
#runs='0 1 2 3 4'
#for run in $runs
#do
#  dataset=$dataset_path$run.pickle
#  python main_deep_learning.py $dataset --supervised --pretrained --posteriors --mbert --log-file $logfile
#done

dataset=/home/moreo/CLESA/rcv2/rcv1-2_doclist_full_processed.pickle
logfile=../log/log_FunBert_fullrcv_static.csv

python main_deep_learning.py $dataset --supervised --pretrained --posteriors --mbert --log-file $logfile