# Lightning modules, see https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from transformers import AdamW
import pytorch_lightning as pl
from models.helpers import init_embeddings
from util.pl_metrics import CustomF1, CustomK
from util.common import define_pad_length, pad


class RecurrentModel(pl.LightningModule):
    def __init__(self, lPretrained, langs, output_size, hidden_size, lVocab_size, learnable_length,
                 drop_embedding_range, drop_embedding_prop, gpus=None):
        """

        :param lPretrained:
        :param langs:
        :param output_size:
        :param hidden_size:
        :param lVocab_size:
        :param learnable_length:
        :param drop_embedding_range:
        :param drop_embedding_prop:
        :param gpus:
        """
        super().__init__()
        self.gpus = gpus
        self.langs = langs
        self.lVocab_size = lVocab_size
        self.learnable_length = learnable_length
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.drop_embedding_range = drop_embedding_range
        self.drop_embedding_prop = drop_embedding_prop
        self.loss = torch.nn.BCEWithLogitsLoss()

        self.microF1 = CustomF1(num_classes=output_size, average='micro', device=self.gpus)
        self.macroF1 = CustomF1(num_classes=output_size, average='macro', device=self.gpus)
        self.microK = CustomK(num_classes=output_size, average='micro', device=self.gpus)
        self.macroK = CustomK(num_classes=output_size, average='macro', device=self.gpus)
        # Language specific metrics - I am not really sure if they should be initialized
        # independently or we can use the metrics init above... # TODO: check it
        self.lang_macroF1 = CustomF1(num_classes=output_size, average='macro', device=self.gpus)
        self.lang_microF1 = CustomF1(num_classes=output_size, average='micro', device=self.gpus)
        self.lang_macroK = CustomF1(num_classes=output_size, average='macro', device=self.gpus)
        self.lang_microK = CustomF1(num_classes=output_size, average='micro', device=self.gpus)

        self.lPretrained_embeddings = nn.ModuleDict()
        self.lLearnable_embeddings = nn.ModuleDict()

        self.n_layers = 1
        self.n_directions = 1
        self.dropout = nn.Dropout(0.6)

        lstm_out = 256
        ff1 = 512
        ff2 = 256

        lpretrained_embeddings = {}
        llearnable_embeddings = {}

        for lang in self.langs:
            pretrained = lPretrained[lang] if lPretrained else None
            pretrained_embeddings, learnable_embeddings, embedding_length = init_embeddings(
                pretrained, self.lVocab_size[lang], self.learnable_length)
            lpretrained_embeddings[lang] = pretrained_embeddings
            llearnable_embeddings[lang] = learnable_embeddings
            self.embedding_length = embedding_length

        self.lPretrained_embeddings.update(lpretrained_embeddings)
        self.lLearnable_embeddings.update(llearnable_embeddings)

        self.rnn = nn.GRU(self.embedding_length, hidden_size)
        self.linear0 = nn.Linear(hidden_size * self.n_directions, lstm_out)
        self.linear1 = nn.Linear(lstm_out, ff1)
        self.linear2 = nn.Linear(ff1, ff2)
        self.label = nn.Linear(ff2, self.output_size)

        # TODO: setting lPretrained to None, letting it to its original value will "bug" first validation
        #  step (i.e., checkpoint will store also its ++ value, I guess, making the saving process too slow)
        lPretrained = None
        self.save_hyperparameters()

    def forward(self, lX):
        l_embed = []
        for lang in sorted(lX.keys()):
            doc_embedding = self.transform(lX[lang], lang)
            l_embed.append(doc_embedding)
        embed = torch.cat(l_embed, dim=0)
        logits = self.label(embed)
        return logits

    def transform(self, X, lang):
        batch_size = X.shape[0]
        X = self.embed(X, lang)
        X = self.embedding_dropout(X, drop_range=self.drop_embedding_range, p_drop=self.drop_embedding_prop,
                                   training=self.training)
        X = X.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size).to(self.device))
        output, _ = self.rnn(X, h_0)
        output = output[-1, :, :]
        output = F.relu(self.linear0(output))
        output = self.dropout(F.relu(self.linear1(output)))
        output = self.dropout(F.relu(self.linear2(output)))
        return output

    def encode(self, lX, l_pad, batch_size=128):
        """
        Returns encoded data (i.e, RNN hidden state at second feed-forward layer - linear1). Dimensionality is 512.
        # TODO: does not run on gpu..
        :param lX:
        :param l_pad:
        :param batch_size:
        :return:
        """
        with torch.no_grad():
            l_embed = {lang: [] for lang in lX.keys()}
            for lang in sorted(lX.keys()):
                for i in range(0, len(lX[lang]), batch_size):
                    if i+batch_size > len(lX[lang]):
                        batch = lX[lang][i:len(lX[lang])]
                    else:
                        batch = lX[lang][i:i+batch_size]
                    max_pad_len = define_pad_length(batch)
                    batch = pad(batch, pad_index=l_pad[lang], max_pad_length=max_pad_len)
                    X = torch.LongTensor(batch)
                    _batch_size = X.shape[0]
                    X = self.embed(X, lang)
                    X = self.embedding_dropout(X, drop_range=self.drop_embedding_range, p_drop=self.drop_embedding_prop,
                                               training=self.training)
                    X = X.permute(1, 0, 2)
                    h_0 = Variable(torch.zeros(self.n_layers * self.n_directions, _batch_size, self.hidden_size).to(self.device))
                    output, _ = self.rnn(X, h_0)
                    output = output[-1, :, :]
                    output = F.relu(self.linear0(output))
                    output = self.dropout(F.relu(self.linear1(output)))
                    l_embed[lang].append(output)
            for k, v in l_embed.items():
                l_embed[k] = torch.cat(v, dim=0).cpu().numpy()
            return l_embed

    def training_step(self, train_batch, batch_idx):
        lX, ly = train_batch
        logits = self.forward(lX)
        _ly = []
        for lang in sorted(lX.keys()):
            _ly.append(ly[lang])
        y = torch.cat(_ly, dim=0)
        loss = self.loss(logits, y)
        # Squashing logits through Sigmoid in order to get confidence score
        predictions = torch.sigmoid(logits) > 0.5
        microF1 = self.microF1(predictions, y)
        macroF1 = self.macroF1(predictions, y)
        microK = self.microK(predictions, y)
        macroK = self.macroK(predictions, y)
        self.log('train-loss', loss,         on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-macroF1', macroF1,   on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-microF1', microF1,   on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-macroK', macroK,     on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-microK', microK,     on_step=True, on_epoch=True, prog_bar=False, logger=True)
        re_lX = self._reconstruct_dict(predictions, ly)
        return {'loss': loss, 'pred': re_lX, 'target': ly}

    def _reconstruct_dict(self, X, ly):
        reconstructed = {}
        _start = 0
        for lang in sorted(ly.keys()):
            lang_batchsize = len(ly[lang])
            reconstructed[lang] = X[_start:_start+lang_batchsize]
            _start += lang_batchsize
        return reconstructed
    
    def training_epoch_end(self, outputs):
        # outputs is a of n dicts of m elements, where n is equal to the number of epoch steps and m is batchsize.
        # here we save epoch level metric values and compute them specifically for each language
        # TODO: this is horrible...
        res_macroF1 = {lang: [] for lang in self.langs}
        res_microF1 = {lang: [] for lang in self.langs}
        res_macroK = {lang: [] for lang in self.langs}
        res_microK = {lang: [] for lang in self.langs}
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
        for lang in self.langs:
            avg_macroF1 = torch.mean(torch.Tensor(res_macroF1[lang]))
            avg_microF1 = torch.mean(torch.Tensor(res_microF1[lang]))
            avg_macroK = torch.mean(torch.Tensor(res_macroK[lang]))
            avg_microK = torch.mean(torch.Tensor(res_microK[lang]))
            self.logger.experiment.add_scalars('train-langs-macroF1', {f'{lang}': avg_macroF1}, self.current_epoch)
            self.logger.experiment.add_scalars('train-langs-microF1', {f'{lang}': avg_microF1}, self.current_epoch)
            self.logger.experiment.add_scalars('train-langs-macroK', {f'{lang}': avg_macroK}, self.current_epoch)
            self.logger.experiment.add_scalars('train-langs-microK', {f'{lang}': avg_microK}, self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        lX, ly = val_batch
        logits = self.forward(lX)
        _ly = []
        for lang in sorted(lX.keys()):
            _ly.append(ly[lang])
        ly = torch.cat(_ly, dim=0)
        loss = self.loss(logits, ly)
        predictions = torch.sigmoid(logits) > 0.5
        microF1 = self.microF1(predictions, ly)
        macroF1 = self.macroF1(predictions, ly)
        microK = self.microK(predictions, ly)
        macroK = self.macroK(predictions, ly)
        self.log('val-loss', loss,         on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val-macroF1', macroF1,   on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val-microF1', microF1,   on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val-macroK', macroK,     on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val-microK', microK,     on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def test_step(self, test_batch, batch_idx):
        lX, ly = test_batch
        logits = self.forward(lX)
        _ly = []
        for lang in sorted(lX.keys()):
            _ly.append(ly[lang])
        ly = torch.cat(_ly, dim=0)
        predictions = torch.sigmoid(logits) > 0.5
        microF1 = self.microF1(predictions, ly)
        macroF1 = self.macroF1(predictions, ly)
        microK = self.microK(predictions, ly)
        macroK = self.macroK(predictions, ly)
        self.log('test-macroF1', macroF1,    on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test-microF1', microF1,    on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test-macroK', macroK,      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test-microK', microK,      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return

    def embed(self, X, lang):
        input_list = []
        if self.lPretrained_embeddings[lang]:
            input_list.append(self.lPretrained_embeddings[lang](X))
        if self.lLearnable_embeddings[lang]:
            input_list.append(self.lLearnable_embeddings[lang](X))
        return torch.cat(tensors=input_list, dim=2)

    def embedding_dropout(self, X, drop_range, p_drop=0.5, training=True):
        if p_drop > 0 and training and drop_range is not None:
            p = p_drop
            drop_from, drop_to = drop_range
            m = drop_to - drop_from  # length of the supervised embedding
            l = X.shape[2]  # total embedding length
            corr = (1 - p)
            X[:, :, drop_from:drop_to] = corr * F.dropout(X[:, :, drop_from:drop_to], p=p)
            X /= (1 - (p * m / l))
        return X

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
        return [optimizer], [scheduler]
