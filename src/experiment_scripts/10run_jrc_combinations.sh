dataset_path=/home/moreo/CLESA/jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_noparallel_processed_run
logfile=./results/funnelling_10run_jrc_CIKM.csv

runs='6 7 8 9' #0 1 2 3 4 5
for run in $runs
do
  dataset=$dataset_path$run.pickle
  #python main_multimodal_cls.py $dataset -o $logfile -P -U -S -c -r -z --l2 --allprob # last combination for CIKM 3 Pr(views) concatenated  (done up to run5)
  python main_multimodal_cls.py $dataset -o $logfile -P -U -S -c -r -z --l2 --allprob # last combination for CIKM 3 views concatenated
  #python main_multimodal_cls.py $dataset -o $logfile -P -U -S -c -r -a -z --l2 --allprob
  #python main_multimodal_cls.py $dataset -o $logfile -P -U -c -r -a -z --l2 --allprob
  #python main_multimodal_cls.py $dataset -o $logfile -P -S -c -r -z --l2 --allprob
  #python main_multimodal_cls.py $dataset -o $logfile -P -U -c -r -z --l2 --allprob
  #python main_multimodal_cls.py $dataset -o $logfile -c -P -U -r -z --l2
  #python main_multimodal_cls.py $dataset -o $logfile -c -P -U -S -r -z --l2
done