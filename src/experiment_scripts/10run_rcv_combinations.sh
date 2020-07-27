dataset_path=/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run
logfile=./results/funnelling_10run_rcv_CIKM_allprob_concatenated.csv

runs='0 1 2 3 4 5 6 7 8 9'
for run in $runs
do
  dataset=$dataset_path$run.pickle
  #python main_multimodal_cls.py $dataset -o $logfile -P -U -S -c -r -z --l2 --allprob # last combination for CIKM 3 Pr(views) concatenated
  python main_multimodal_cls.py $dataset -o $logfile -P -U -S -c -r -z --l2 --allprob # last combination for CIKM 3 views concatenated
  #python main_multimodal_cls.py $dataset -o $logfile -P -U -c -r -a -z --l2 --allprob
  #python main_multimodal_cls.py $dataset -o $logfile -P -U -S -c -r -a -z --l2 --allprob
  #python main_multimodal_cls.py $dataset -o $logfile -P -S -c -r -z --l2 --allprob
  #python main_multimodal_cls.py $dataset -o $logfile -P -U -c -r -z --l2 --allprob
  #python main_multimodal_cls.py $dataset -o $logfile -c -P -U -r -z --l2
  #python main_multimodal_cls.py $dataset -o $logfile -c -P -U -S -r -z --l2
done