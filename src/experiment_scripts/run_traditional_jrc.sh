#!/usr/bin/env bash

dataset=/home/moreo/CLESA/jrc_acquis/jrc_doclist_1958-2005vs2006_all_top300_noparallel_processed_run0.pickle

######################################## POSTERIORS
                                                                                  # Posteriors
python main_multimodal_cls.py $dataset -P                                         # + zscore
python main_multimodal_cls.py $dataset -P -z                                      # +l2norm
python main_multimodal_cls.py $dataset -P -z --l2                                 # +feature weight


######################################### WCE
                                                                                  #WCE supervised
python main_multimodal_cls.py $dataset -S                                         # + zscore
python main_multimodal_cls.py $dataset -S -z                                      # +l2norm
python main_multimodal_cls.py $dataset -S -z --l2                                 # +feature weight
python main_multimodal_cls.py $dataset -S -z -r --l2                               # + SIF - PCA

python main_multimodal_cls.py $dataset -S -z -p 250 --l2                           # +feature weight + pca
python main_multimodal_cls.py $dataset -S -z -r -p 250 --l2                        # + SIF

python main_multimodal_cls.py $dataset -S -z --l2 --feat-weight ig                # -feature weight
python main_multimodal_cls.py $dataset -S -z -r --l2 --feat-weight ig
python main_multimodal_cls.py $dataset -S -z -p 250 --l2 --feat-weight ig           # + pca
python main_multimodal_cls.py $dataset -S -z -r -p 250 --l2 --feat-weight ig


python main_multimodal_cls.py $dataset -S -z --l2 --feat-weight pmi
python main_multimodal_cls.py $dataset -S -z -r --l2 --feat-weight pmi
python main_multimodal_cls.py $dataset -S -z -p 250 --l2 --feat-weight pmi
python main_multimodal_cls.py $dataset -S -z -r -p 250 --l2 --feat-weight pmi

################################# MUSE

                                                                                  # MUSE unsupervised
python main_multimodal_cls.py $dataset -U                                         # + zscore
python main_multimodal_cls.py $dataset -U -z                                      # +l2norm
python main_multimodal_cls.py $dataset -U -z --l2                                 # +feature weight
python main_multimodal_cls.py $dataset -U -z -r --l2                              # + SIF - PCA

python main_multimodal_cls.py $dataset -U -z --l2 --feat-weight ig                # -feature weight + pca
python main_multimodal_cls.py $dataset -U -z -r --l2 --feat-weight ig

python main_multimodal_cls.py $dataset -U -z --l2 --feat-weight pmi
python main_multimodal_cls.py $dataset -U -z -r --l2 --feat-weight pmi
