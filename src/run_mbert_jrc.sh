#!/usr/bin/env bash

#dataset_path=/home/moreo/CLESA/jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_noparallel_processed_run
#logfile=../log/log_mBert_jrc_NEW.csv
#
#runs='0 1 2 3 4'
#for run in $runs
#do
#  dataset=$dataset_path$run.pickle
#  python main_mbert.py --dataset $dataset --log-file $logfile --nepochs=50
#done

logfile=../log/log_mBert_fulljrc.csv
dataset=/home/moreo/CLESA/jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_full_processed.pickle
python main_mbert.py --dataset $dataset --log-file $logfile --nepochs=50