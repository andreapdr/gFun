# Lightning modules, see https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
import torch
from torch import nn
from transformers import AdamW
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.metrics import F1, Accuracy
from torch.optim.lr_scheduler import StepLR
from models.helpers import init_embeddings
from util.pl_metrics import CustomF1
from util.evaluation import evaluate


class RecurrentModel(pl.LightningModule):
    """
    Check out for logging insight https://www.learnopencv.com/tensorboard-with-pytorch-lightning/
    """

    def __init__(self, lPretrained, langs, output_size, hidden_size, lVocab_size, learnable_length,
                 drop_embedding_range, drop_embedding_prop, gpus=None):
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

        self.accuracy = Accuracy()
        self.microF1_tr = CustomF1(num_classes=output_size, average='micro', device=self.gpus)
        self.macroF1_tr = CustomF1(num_classes=output_size, average='macro', device=self.gpus)
        self.microF1_va = CustomF1(num_classes=output_size, average='micro', device=self.gpus)
        self.macroF1_va = CustomF1(num_classes=output_size, average='macro', device=self.gpus)
        self.microF1_te = CustomF1(num_classes=output_size, average='micro', device=self.gpus)
        self.macroF1_te = CustomF1(num_classes=output_size, average='macro', device=self.gpus)

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

        lPretrained = None  # TODO: setting lPretrained to None, letting it to its original value will bug first
        #  validation step (i.e., checkpoint will store also its ++ value, I guess, making the saving process too slow)
        self.save_hyperparameters()

    def forward(self, lX):
        _tmp = []
        for lang in sorted(lX.keys()):
            doc_embedding = self.transform(lX[lang], lang)
            _tmp.append(doc_embedding)
        embed = torch.cat(_tmp, dim=0)
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

    def training_step(self, train_batch, batch_idx):
        lX, ly = train_batch
        logits = self.forward(lX)
        _ly = []
        for lang in sorted(lX.keys()):
            _ly.append(ly[lang])
        ly = torch.cat(_ly, dim=0)
        loss = self.loss(logits, ly)
        # Squashing logits through Sigmoid in order to get confidence score
        predictions = torch.sigmoid(logits) > 0.5
        accuracy = self.accuracy(predictions, ly)
        microF1 = self.microF1_tr(predictions, ly)
        macroF1 = self.macroF1_tr(predictions, ly)
        self.log('train-loss', loss,         on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-macroF1', macroF1,   on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train-microF1', microF1,   on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        lX, ly = val_batch
        logits = self.forward(lX)
        _ly = []
        for lang in sorted(lX.keys()):
            _ly.append(ly[lang])
        ly = torch.cat(_ly, dim=0)
        loss = self.loss(logits, ly)
        predictions = torch.sigmoid(logits) > 0.5
        accuracy = self.accuracy(predictions, ly)
        microF1 = self.microF1_va(predictions, ly)
        macroF1 = self.macroF1_va(predictions, ly)
        self.log('val-loss', loss,         on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val-accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val-macroF1', macroF1,   on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val-microF1', microF1,   on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def test_step(self, test_batch, batch_idx):
        lX, ly = test_batch
        logits = self.forward(lX)
        _ly = []
        for lang in sorted(lX.keys()):
            _ly.append(ly[lang])
        ly = torch.cat(_ly, dim=0)
        predictions = torch.sigmoid(logits) > 0.5
        accuracy = self.accuracy(predictions, ly)
        microF1 = self.microF1_te(predictions, ly)
        macroF1 = self.macroF1_te(predictions, ly)
        self.log('test-accuracy', accuracy,  on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test-macroF1', macroF1,    on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test-microF1', microF1,    on_step=False, on_epoch=True, prog_bar=False, logger=True)
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
