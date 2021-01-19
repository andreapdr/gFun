import torch
from torch import nn
from torch.optim import Adam
from transformers import AdamW
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.metrics import F1, Accuracy, Metric
from torch.optim.lr_scheduler import StepLR

from util.evaluation import evaluate
from typing import Any, Optional, Tuple
from pytorch_lightning.metrics.utils import _input_format_classification_one_hot, class_reduce
import numpy as np


def init_embeddings(pretrained, vocab_size, learnable_length):
    """
    Compute the embedding matrix
    :param pretrained:
    :param vocab_size:
    :param learnable_length:
    :return:
    """
    pretrained_embeddings = None
    pretrained_length = 0
    if pretrained is not None:
        pretrained_length = pretrained.shape[1]
        assert pretrained.shape[0] == vocab_size, 'pre-trained matrix does not match with the vocabulary size'
        pretrained_embeddings = nn.Embedding(vocab_size, pretrained_length)
        # requires_grad=False sets the embedding layer as NOT trainable
        pretrained_embeddings.weight = nn.Parameter(pretrained, requires_grad=False)

    learnable_embeddings = None
    if learnable_length > 0:
        learnable_embeddings = nn.Embedding(vocab_size, learnable_length)

    embedding_length = learnable_length + pretrained_length
    assert embedding_length > 0, '0-size embeddings'
    return pretrained_embeddings, learnable_embeddings, embedding_length


class RecurrentModel(pl.LightningModule):
    """
    Check out for logging insight https://www.learnopencv.com/tensorboard-with-pytorch-lightning/
    """

    def __init__(self, lPretrained, langs, output_size, hidden_size, lVocab_size, learnable_length,
                 drop_embedding_range, drop_embedding_prop, lMuse_debug=None, multilingual_index_debug=None):
        super().__init__()
        self.langs = langs
        self.lVocab_size = lVocab_size
        self.learnable_length = learnable_length
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.drop_embedding_range = drop_embedding_range
        self.drop_embedding_prop = drop_embedding_prop
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.microf1 = F1(num_classes=output_size, multilabel=True, average='micro')
        self.macrof1 = F1(num_classes=output_size, multilabel=True, average='macro')
        self.accuracy = Accuracy()
        self.customMetrics = CustomMetrics(num_classes=output_size, multilabel=True, average='micro')

        self.lPretrained_embeddings = nn.ModuleDict()
        self.lLearnable_embeddings = nn.ModuleDict()

        self.n_layers = 1
        self.n_directions = 1
        self.dropout = nn.Dropout(0.6)

        # TODO: debug setting
        self.lMuse = lMuse_debug
        self.multilingual_index_debug = multilingual_index_debug

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
        # TODO: double check StepLR scheduler...
        lX, ly = train_batch
        logits = self.forward(lX)
        _ly = []
        for lang in sorted(lX.keys()):
            _ly.append(ly[lang])
        ly = torch.cat(_ly, dim=0)
        loss = self.loss(logits, ly)

        # Squashing logits through Sigmoid in order to get confidence score
        predictions = torch.sigmoid(logits) > 0.5

        # microf1 = self.microf1(predictions, ly)
        # macrof1 = self.macrof1(predictions, ly)
        accuracy = self.accuracy(predictions, ly)
        # l_pred = {lang: predictions.detach().cpu().numpy()}
        # l_labels = {lang: ly.detach().cpu().numpy()}
        # l_eval = evaluate(l_labels, l_pred, n_jobs=1)

        self.log('train-loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train-accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        lX, ly = val_batch
        logits = self.forward(lX)
        _ly = []
        for lang in sorted(lX.keys()):
            _ly.append(ly[lang])
        ly = torch.cat(_ly, dim=0)
        loss = self.loss(logits, ly)
        predictions = torch.sigmoid(logits) > 0.5
        # microf1 = self.microf1(predictions, ly)
        # macrof1 = self.macrof1(predictions, ly)
        accuracy = self.accuracy(predictions, ly)

        # l_pred = {lang: predictions.detach().cpu().numpy()}
        # l_labels = {lang: y.detach().cpu().numpy()}
        # l_eval = evaluate(l_labels, l_pred, n_jobs=1)

        self.log('val-loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val-accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return

    def test_step(self, test_batch, batch_idx):
        lX, ly = test_batch
        logits = self.forward(lX)
        _ly = []
        for lang in sorted(lX.keys()):
            _ly.append(ly[lang])
        ly = torch.cat(_ly, dim=0)
        predictions = torch.sigmoid(logits) > 0.5
        accuracy = self.accuracy(predictions, ly)
        custom_metric = self.customMetrics(logits, ly)  # TODO
        self.log('test-accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test-custom', custom_metric, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {'pred': predictions, 'target': ly}

    def test_epoch_end(self, outputs):
        # all_pred = torch.vstack([out['pred'] for out in outputs])   # TODO
        # all_y = torch.vstack([out['target'] for out in outputs])    # TODO
        # r = eval(all_y, all_pred)
        # print(r)
        # X = torch.cat(X).view([X[0].shape[0], len(X)])
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


class CustomMetrics(Metric):
    def __init__(
            self,
            num_classes: int,
            beta: float = 1.0,
            threshold: float = 0.5,
            average: str = "micro",
            multilabel: bool = False,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step, process_group=process_group,
        )

        self.num_classes = num_classes
        self.beta = beta
        self.threshold = threshold
        self.average = average
        self.multilabel = multilabel

        allowed_average = ("micro", "macro", "weighted", None)
        if self.average not in allowed_average:
            raise ValueError('Argument `average` expected to be one of the following:'
                             f' {allowed_average} but got {self.average}')

        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("predicted_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("actual_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        true_positives, predicted_positives, actual_positives = _fbeta_update(
            preds, target, self.num_classes, self.threshold, self.multilabel
        )

        self.true_positives += true_positives
        self.predicted_positives += predicted_positives
        self.actual_positives += actual_positives

    def compute(self):
        """
        Computes metrics over state.
        """
        return _fbeta_compute(self.true_positives, self.predicted_positives,
                              self.actual_positives, self.beta, self.average)


def _fbeta_update(
        preds: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        threshold: float = 0.5,
        multilabel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds, target = _input_format_classification_one_hot(
        num_classes, preds, target, threshold, multilabel
    )
    true_positives = torch.sum(preds * target, dim=1)
    predicted_positives = torch.sum(preds, dim=1)
    actual_positives = torch.sum(target, dim=1)
    return true_positives, predicted_positives, actual_positives


def _fbeta_compute(
        true_positives: torch.Tensor,
        predicted_positives: torch.Tensor,
        actual_positives: torch.Tensor,
        beta: float = 1.0,
        average: str = "micro"
) -> torch.Tensor:
    if average == "micro":
        precision = true_positives.sum().float() / predicted_positives.sum()
        recall = true_positives.sum().float() / actual_positives.sum()
    else:
        precision = true_positives.float() / predicted_positives
        recall = true_positives.float() / actual_positives

    num = (1 + beta ** 2) * precision * recall
    denom = beta ** 2 * precision + recall
    new_num = 2 * true_positives
    new_fp = predicted_positives - true_positives
    new_fn = actual_positives - true_positives
    new_den = 2 * true_positives + new_fp + new_fn
    if new_den.sum() == 0:
        # whats is the correct return type ? TODO
        return 1.
    return class_reduce(num, denom, weights=actual_positives, class_reduction=average)
