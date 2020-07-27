dataset=/home/moreo/CLESA/jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_noparallel_processed_run0.pickle
logfile=./results/10run_jrc_final_results.csv

runs='0 1 2 3 4 5 6 7 8 9'
for run in $runs
do
  dataset=$dataset_path$run.pickle
  python main_multimodal_cls.py $dataset -o $logfile -P -z -c --l2
  python main_multimodal_cls.py $dataset -o $logfile -S -z -c --l2
  python main_multimodal_cls.py $dataset -o $logfile -U -z -c --l2

done
