#!/usr/bin/env bash

#dataset_path=/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run
#logfile=../log/log_mBert_rcv_NEW.csv
#
#runs='0 1 2 3 4'
#for run in $runs
#do
#  dataset=$dataset_path$run.pickle
#  python main_mbert.py --dataset $dataset --log-file $logfile --nepochs=50
#done

logfile=../log/log_mBert_fullrcv.csv
dataset=/home/moreo/CLESA/rcv2/rcv1-2_doclist_full_processed.pickle
python main_mbert.py --dataset $dataset --log-file $logfile --nepochs=30 --patience 3