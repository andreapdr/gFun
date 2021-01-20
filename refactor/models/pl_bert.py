import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from transformers import BertForSequenceClassification, AdamW
from pytorch_lightning.metrics import Accuracy
from util.pl_metrics import CustomF1


class BertModel(pl.LightningModule):

    def __init__(self, output_size, stored_path, gpus=None):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.gpus = gpus
        self.accuracy = Accuracy()
        self.microF1_tr = CustomF1(num_classes=output_size, average='micro', device=self.gpus)
        self.macroF1_tr = CustomF1(num_classes=output_size, average='macro', device=self.gpus)
        self.microF1_va = CustomF1(num_classes=output_size, average='micro', device=self.gpus)
        self.macroF1_va = CustomF1(num_classes=output_size, average='macro', device=self.gpus)
        self.microF1_te = CustomF1(num_classes=output_size, average='micro', device=self.gpus)
        self.macroF1_te = CustomF1(num_classes=output_size, average='macro', device=self.gpus)

        if stored_path:
            self.bert = BertForSequenceClassification.from_pretrained(stored_path,
                                                                      num_labels=output_size,
                                                                      output_hidden_states=True)
        else:
            self.bert = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',
                                                                      num_labels=output_size,
                                                                      output_hidden_states=True)
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
        # Squashing logits through Sigmoid in order to get confidence score
        predictions = torch.sigmoid(logits) > 0.5
        accuracy = self.accuracy(predictions, y)
        microF1 = self.microF1_tr(predictions, y)
        macroF1 = self.macroF1_tr(predictions, y)
        self.log('train-loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-macroF1', macroF1, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-microF1', microF1, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        X, y, _, batch_langs = val_batch
        X = torch.cat(X).view([X[0].shape[0], len(X)])
        y = y.type(torch.cuda.FloatTensor)
        logits, _ = self.forward(X)
        loss = self.loss(logits, y)
        predictions = torch.sigmoid(logits) > 0.5
        accuracy = self.accuracy(predictions, y)
        microF1 = self.microF1_va(predictions, y)
        macroF1 = self.macroF1_va(predictions, y)
        self.log('val-loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val-accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val-macroF1', macroF1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val-microF1', microF1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    # def test_step(self, test_batch, batch_idx):
    #     lX, ly = test_batch
    #     logits = self.forward(lX)
    #     _ly = []
    #     for lang in sorted(lX.keys()):
    #         _ly.append(ly[lang])
    #     ly = torch.cat(_ly, dim=0)
    #     predictions = torch.sigmoid(logits) > 0.5
    #     accuracy = self.accuracy(predictions, ly)
    #     microF1 = self.microF1_te(predictions, ly)
    #     macroF1 = self.macroF1_te(predictions, ly)
    #     self.log('test-accuracy', accuracy,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
    #     self.log('test-macroF1', macroF1,    on_step=False, on_epoch=True, prog_bar=False, logger=True)
    #     self.log('test-microF1', microF1,    on_step=False, on_epoch=True, prog_bar=False, logger=True)
    #     return

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
