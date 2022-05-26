#adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
import torch
from transformers import BertForSequenceClassification
from time import time
from src.util.file import create_if_not_exist
import warnings

class OldEarlyStopping:

    def __init__(self, model, optimizer, patience=20, verbose=True, checkpoint='./checkpoint.pt', is_bert=False):
        # set patience to 0 or -1 to avoid stopping, but still keeping track of the best value and model parameters
        self.patience_limit = patience
        self.patience = patience
        self.verbose = verbose
        self.best_score = None
        self.best_epoch = None
        self.stop_time  = None
        self.checkpoint = checkpoint
        self.model = model
        self.optimizer = optimizer
        self.STOP = False
        self.is_bert = is_bert

    def __call__(self, watch_score, epoch):

        if self.STOP:
            return

        if self.best_score is None or watch_score >= self.best_score:
            self.best_score = watch_score
            self.best_epoch = epoch
            self.stop_time = time()
            if self.checkpoint:
                self.print(f'[early-stop] improved, saving model in {self.checkpoint}')
                if self.is_bert:
                    print(f'Serializing Huggingface model...')
                    create_if_not_exist(self.checkpoint)
                    self.model.save_pretrained(self.checkpoint)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        torch.save(self.model, self.checkpoint)
                        # with open(self.checkpoint)
                        # torch.save({'state_dict': self.model.state_dict(),
                        #             'optimizer_state_dict': self.optimizer.state_dict()}, self.checkpoint)
            else:
                self.print(f'[early-stop] improved')
            self.patience = self.patience_limit
        else:
            self.patience -= 1
            if self.patience == 0:
                self.STOP = True
                self.print(f'[early-stop] patience exhausted')
            else:
                if self.patience>0: # if negative, then early-stop is ignored
                    self.print(f'[early-stop] patience={self.patience}')

    def reinit_counter(self):
        self.STOP = False
        self.patience=self.patience_limit

    def restore_checkpoint(self):
        print(f'restoring best model from epoch {self.best_epoch}...')
        if self.is_bert:
            return BertForSequenceClassification.from_pretrained(self.checkpoint)
        else:
            return torch.load(self.checkpoint)

    def print(self, msg):
        if self.verbose:
            print(msg)
