import os

import numpy as np
import pandas as pd


class CSVlog:
    def __init__(self, file, autoflush=True, verbose=False):
        self.file = file
        self.columns = ['method',
                        'setting',
                        'optimc',
                        'sif',
                        'zscore',
                        'l2',
                        'dataset',
                        'time_tr',
                        'time_te',
                        'lang',
                        'macrof1',
                        'microf1',
                        'macrok',
                        'microk',
                        'macrop',
                        'microp',
                        'macror',
                        'micror',
                        'notes']
        self.autoflush = autoflush
        self.verbose = verbose
        if os.path.exists(file):
            self.tell('Loading existing file from {}'.format(file))
            self.df = pd.read_csv(file, sep='\t')
        else:
            self.tell('File {} does not exist. Creating new frame.'.format(file))
            dir = os.path.dirname(self.file)
            if dir and not os.path.exists(dir): os.makedirs(dir)
            self.df = pd.DataFrame(columns=self.columns)

    def already_calculated(self, id):
        return (self.df['id'] == id).any()

    def add_row(self, method, setting, optimc, sif, zscore, l2, dataset, time_tr, time_te, lang,
                macrof1, microf1, macrok=np.nan, microk=np.nan, macrop=np.nan, microp=np.nan, macror=np.nan, micror=np.nan,  notes=''):
        s = pd.Series([method, setting, optimc, sif, zscore, l2, dataset, time_tr, time_te, lang,
                       macrof1, microf1, macrok, macrop, microp, macror, micror, microk, notes],
                      index=self.columns)
        self.df = self.df.append(s, ignore_index=True)
        if self.autoflush: self.flush()
        self.tell(s.to_string())

    def flush(self):
        self.df.to_csv(self.file, index=False, sep='\t')

    def tell(self, msg):
        if self.verbose:
            print(msg)
