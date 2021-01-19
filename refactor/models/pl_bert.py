import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig
from pytorch_lightning.metrics import F1, Accuracy, Metric


class BertModel(pl.LightningModule):

    def __init__(self, output_size, stored_path):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        if stored_path:
            self.bert = BertForSequenceClassification.from_pretrained(stored_path,
                                                                      num_labels=output_size,
                                                                      output_hidden_states=True)
        else:
            self.bert = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',
                                                                      num_labels=output_size,
                                                                      output_hidden_states=True)
        self.accuracy = Accuracy()
        self.save_hyperparameters()

    def forward(self, X):
        logits = self.bert(X)
        return logits

    def training_step(self, train_batch, batch_idx):
        X, y, _, batch_langs = train_batch
        X = torch.cat(X).view([X[0].shape[0], len(X)])
        y = y.type(torch.cuda.FloatTensor)
        logits, _ = self.forward(X)
        loss = self.loss(logits, y)
        predictions = torch.sigmoid(logits) > 0.5
        accuracy = self.accuracy(predictions, y)
        self.log('train-loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train-accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y, _, batch_langs = val_batch
        X = torch.cat(X).view([X[0].shape[0], len(X)])
        y = y.type(torch.cuda.FloatTensor)
        logits, _ = self.forward(X)
        loss = self.loss(logits, y)
        predictions = torch.sigmoid(logits) > 0.5
        accuracy = self.accuracy(predictions, y)
        self.log('val-loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val-accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return

    def configure_optimizers(self, lr=3e-5, weight_decay=0.01):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.bert.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.bert.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
        return [optimizer], [scheduler]
