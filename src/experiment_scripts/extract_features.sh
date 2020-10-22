#!/usr/bin/env bash

dataset_path=/home/moreo/CLESA/jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_noparallel_processed_run#

runs='1 2 3 4 5 6 7 8 9'
for run in $runs
do
  dataset=$dataset_path$run.pickle
  modelpath=/home/andreapdr/funneling_pdr/hug_checkpoint/mBERT-jrc_run$runs
  python main_mbert_extractor.py --dataset $dataset --modelpath $modelpath
done

dataset=/home/moreo/CLESA/jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_full_processed.pickle
python main_mbert_extractor.py --dataset $dataset --modelpath $modelpath