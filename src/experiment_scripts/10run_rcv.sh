dataset_path=/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run
logfile=./results/10run_rcv_final_results.csv

runs='0 1 2 3 4 5 6 7 8 9'

for run in $runs
do
  dataset=$dataset_path$run.pickle
  python main_multimodal_cls.py $dataset -o $logfile -P -z -c --l2
  python main_multimodal_cls.py $dataset -o $logfile -S -z -c --l2
  python main_multimodal_cls.py $dataset -o $logfile -U -z -c --l2

done


