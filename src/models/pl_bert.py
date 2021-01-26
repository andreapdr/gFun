import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import StepLR
from transformers import BertForSequenceClassification, AdamW

from src.util.common import define_pad_length, pad
from src.util.pl_metrics import CustomF1, CustomK


class BertModel(pl.LightningModule):

    def __init__(self, output_size, stored_path, gpus=None):
        """
        Init Bert model.
        :param output_size:
        :param stored_path:
        :param gpus:
        """
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.gpus = gpus
        self.microF1 = CustomF1(num_classes=output_size, average='micro', device=self.gpus)
        self.macroF1 = CustomF1(num_classes=output_size, average='macro', device=self.gpus)
        self.microK = CustomK(num_classes=output_size, average='micro', device=self.gpus)
        self.macroK = CustomK(num_classes=output_size, average='macro', device=self.gpus)
        # Language specific metrics to compute metrics at epoch level
        self.lang_macroF1 = CustomF1(num_classes=output_size, average='macro', device=self.gpus)
        self.lang_microF1 = CustomF1(num_classes=output_size, average='micro', device=self.gpus)
        self.lang_macroK = CustomF1(num_classes=output_size, average='macro', device=self.gpus)
        self.lang_microK = CustomF1(num_classes=output_size, average='micro', device=self.gpus)

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
        y = y.type(torch.FloatTensor)
        y = y.to('cuda' if self.gpus else 'cpu')
        logits, _ = self.forward(X)
        loss = self.loss(logits, y)
        # Squashing logits through Sigmoid in order to get confidence score
        predictions = torch.sigmoid(logits) > 0.5
        microF1 = self.microF1(predictions, y)
        macroF1 = self.macroF1(predictions, y)
        microK = self.microK(predictions, y)
        macroK = self.macroK(predictions, y)
        self.log('train-loss', loss,        on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-macroF1', macroF1,  on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-microF1', microF1,  on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-macroK', macroK,    on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-microK', microK,    on_step=True, on_epoch=True, prog_bar=False, logger=True)
        lX, ly = self._reconstruct_dict(predictions, y, batch_langs)
        return {'loss': loss, 'pred': lX, 'target': ly}

    def training_epoch_end(self, outputs):
        langs = []
        for output in outputs:
            langs.extend(list(output['pred'].keys()))
        langs = set(langs)
        # outputs is a of n dicts of m elements, where n is equal to the number of epoch steps and m is batchsize.
        # here we save epoch level metric values and compute them specifically for each language
        res_macroF1 = {lang: [] for lang in langs}
        res_microF1 = {lang: [] for lang in langs}
        res_macroK = {lang: [] for lang in langs}
        res_microK = {lang: [] for lang in langs}
        for output in outputs:
            lX, ly = output['pred'], output['target']
            for lang in lX.keys():
                X, y = lX[lang], ly[lang]
                lang_macroF1 = self.lang_macroF1(X, y)
                lang_microF1 = self.lang_microF1(X, y)
                lang_macroK = self.lang_macroK(X, y)
                lang_microK = self.lang_microK(X, y)

                res_macroF1[lang].append(lang_macroF1)
                res_microF1[lang].append(lang_microF1)
                res_macroK[lang].append(lang_macroK)
                res_microK[lang].append(lang_microK)
        for lang in langs:
            avg_macroF1 = torch.mean(torch.Tensor(res_macroF1[lang]))
            avg_microF1 = torch.mean(torch.Tensor(res_microF1[lang]))
            avg_macroK = torch.mean(torch.Tensor(res_macroK[lang]))
            avg_microK = torch.mean(torch.Tensor(res_microK[lang]))
            self.logger.experiment.add_scalars('train-langs-macroF1', {f'{lang}': avg_macroF1}, self.current_epoch)
            self.logger.experiment.add_scalars('train-langs-microF1', {f'{lang}': avg_microF1}, self.current_epoch)
            self.logger.experiment.add_scalars('train-langs-macroK', {f'{lang}': avg_macroK}, self.current_epoch)
            self.logger.experiment.add_scalars('train-langs-microK', {f'{lang}': avg_microK}, self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        X, y, _, batch_langs = val_batch
        X = torch.cat(X).view([X[0].shape[0], len(X)])
        y = y.type(torch.FloatTensor)
        y = y.to('cuda' if self.gpus else 'cpu')
        logits, _ = self.forward(X)
        loss = self.loss(logits, y)
        predictions = torch.sigmoid(logits) > 0.5
        microF1 = self.microF1(predictions, y)
        macroF1 = self.macroF1(predictions, y)
        microK = self.microK(predictions, y)
        macroK = self.macroK(predictions, y)
        self.log('val-loss', loss,          on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val-macroF1', macroF1,    on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val-microF1', microF1,    on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val-macroK', macroK,      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val-microK', microK,      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def test_step(self, test_batch, batch_idx):
        X, y, _, batch_langs = test_batch
        X = torch.cat(X).view([X[0].shape[0], len(X)])
        y = y.type(torch.FloatTensor)
        y = y.to('cuda' if self.gpus else 'cpu')
        logits, _ = self.forward(X)
        loss = self.loss(logits, y)
        # Squashing logits through Sigmoid in order to get confidence score
        predictions = torch.sigmoid(logits) > 0.5
        microF1 = self.microF1(predictions, y)
        macroF1 = self.macroF1(predictions, y)
        microK = self.microK(predictions, y)
        macroK = self.macroK(predictions, y)
        self.log('test-macroF1', macroF1,   on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test-microF1', microF1,   on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test-macroK', macroK,     on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test-microK', microK,     on_step=False, on_epoch=True, prog_bar=True, logger=True)
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

    def encode(self, lX, batch_size=64):
        with torch.no_grad():
            l_embed = {lang: [] for lang in lX.keys()}
            for lang in sorted(lX.keys()):
                for i in range(0, len(lX[lang]), batch_size):
                    if i + batch_size > len(lX[lang]):
                        batch = lX[lang][i:len(lX[lang])]
                    else:
                        batch = lX[lang][i:i + batch_size]
                    max_pad_len = define_pad_length(batch)
                    batch = pad(batch, pad_index=self.bert.config.pad_token_id, max_pad_length=max_pad_len)
                    batch = torch.LongTensor(batch).to('cuda' if self.gpus else 'cpu')
                    _, output = self.forward(batch)

                    # deleting batch from gpu to avoid cuda OOM
                    del batch
                    torch.cuda.empty_cache()

                    doc_embeds = output[-1][:, 0, :]
                    l_embed[lang].append(doc_embeds.cpu())
            for k, v in l_embed.items():
                l_embed[k] = torch.cat(v, dim=0).numpy()
            return l_embed

    @staticmethod
    def _reconstruct_dict(predictions, y, batch_langs):
        reconstructed_x = {lang: [] for lang in set(batch_langs)}
        reconstructed_y = {lang: [] for lang in set(batch_langs)}
        for i, pred in enumerate(predictions):
            reconstructed_x[batch_langs[i]].append(pred)
            reconstructed_y[batch_langs[i]].append(y[i])
        for k, v in reconstructed_x.items():
            reconstructed_x[k] = torch.cat(v).view(-1, predictions.shape[1])
        for k, v in reconstructed_y.items():
            reconstructed_y[k] = torch.cat(v).view(-1, predictions.shape[1])
        return reconstructed_x, reconstructed_y
