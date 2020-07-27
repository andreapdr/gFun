#!/usr/bin/env bash

dataset=/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle

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

python main_multimodal_cls.py $dataset -S -z -p 50 --l2                           # +feature weight + pca
python main_multimodal_cls.py $dataset -S -z -r -p 50 --l2                        # + SIF

python main_multimodal_cls.py $dataset -S -z --l2 --feat-weight ig                # -feature weight
python main_multimodal_cls.py $dataset -S -z -r --l2 --feat-weight ig
python main_multimodal_cls.py $dataset -S -z -p 50 --l2 --feat-weight ig           # + pca
python main_multimodal_cls.py $dataset -S -z -r -p 50 --l2 --feat-weight ig


python main_multimodal_cls.py $dataset -S -z --l2 --feat-weight pmi
python main_multimodal_cls.py $dataset -S -z -r --l2 --feat-weight pmi
python main_multimodal_cls.py $dataset -S -z -p 50 --l2 --feat-weight pmi
python main_multimodal_cls.py $dataset -S -z -r -p 50 --l2 --feat-weight pmi

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
