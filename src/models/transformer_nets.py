from copy import deepcopy
import torch
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig
from sklearn.model_selection import train_test_split
from src.util.evaluation import *
from time import time
from src.util.csv_logger import CSVLog


def predict(logits, classification_type='multilabel'):
    if classification_type == 'multilabel':
        prediction = torch.sigmoid(logits) > 0.5
    elif classification_type == 'singlelabel':
        prediction = torch.argmax(logits, dim=1).view(-1, 1)
    else:
        print('unknown classification type')

    return prediction.detach().cpu().numpy()


class TrainingDataset(Dataset):
    """
    data: dict of lang specific tokenized data
    labels: dict of lang specific targets
    """

    def __init__(self, data, labels):
        self.langs = data.keys()
        self.lang_ids = {lang: identifier for identifier, lang in enumerate(self.langs)}

        for i, lang in enumerate(self.langs):
            _data = data[lang]['input_ids']
            _data = np.array(_data)
            _labels = labels[lang]
            _lang_value = np.full(len(_data), self.lang_ids[lang])

            if i == 0:
                self.data = _data
                self.labels = _labels
                self.lang_index = _lang_value
            else:
                self.data = np.vstack((self.data, _data))
                self.labels = np.vstack((self.labels, _labels))
                self.lang_index = np.concatenate((self.lang_index, _lang_value))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        lang = self.lang_index[idx]

        return x, torch.tensor(y, dtype=torch.float), lang

    def get_lang_ids(self):
        return self.lang_ids

    def get_nclasses(self):
        if hasattr(self, 'labels'):
            return len(self.labels[0])
        else:
            print('Method called before init!')


class ExtractorDataset(Dataset):
    """
    data: dict of lang specific tokenized data
    labels: dict of lang specific targets
    """

    def __init__(self, data):
        self.langs = data.keys()
        self.lang_ids = {lang: identifier for identifier, lang in enumerate(self.langs)}

        for i, lang in enumerate(self.langs):
            _data = data[lang]['input_ids']
            _data = np.array(_data)
            _lang_value = np.full(len(_data), self.lang_ids[lang])

            if i == 0:
                self.data = _data
                self.lang_index = _lang_value
            else:
                self.data = np.vstack((self.data, _data))
                self.lang_index = np.concatenate((self.lang_index, _lang_value))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        lang = self.lang_index[idx]

        return x, lang

    def get_lang_ids(self):
        return self.lang_ids


def get_model(n_out):
    print('# Initializing model ...')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=n_out,
                                                          output_hidden_states=True)
    return model


def init_optimizer(model, lr, weight_decay=0):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_tr_val_split(l_tokenized_tr, l_devel_target, val_prop, max_val, seed):
    l_split_va = deepcopy(l_tokenized_tr)
    l_split_val_target = {l: [] for l in l_tokenized_tr.keys()}
    l_split_tr = deepcopy(l_tokenized_tr)
    l_split_tr_target = {l: [] for l in l_tokenized_tr.keys()}

    for lang in l_tokenized_tr.keys():
        val_size = int(min(len(l_tokenized_tr[lang]['input_ids']) * val_prop, max_val))
        l_split_tr[lang]['input_ids'], l_split_va[lang]['input_ids'], l_split_tr_target[lang], l_split_val_target[
            lang] = \
            train_test_split(l_tokenized_tr[lang]['input_ids'], l_devel_target[lang], test_size=val_size,
                             random_state=seed, shuffle=True)

    return l_split_tr, l_split_tr_target, l_split_va, l_split_val_target


def do_tokenization(l_dataset, max_len=512, verbose=True):
    if verbose:
        print('# Starting Tokenization ...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    langs = l_dataset.keys()
    l_tokenized = {}
    for lang in langs:
        l_tokenized[lang] = tokenizer(l_dataset[lang],
                                      truncation=True,
                                      max_length=max_len,
                                      padding='max_length')
    return l_tokenized


def train(model, train_dataloader, epoch, criterion, optim, method_name, tinit, logfile, opt, log_interval=10):
    _dataset_path = opt.dataset.split('/')[-1].split('_')
    dataset_id = _dataset_path[0] + _dataset_path[-1]

    loss_history = []
    model.train()

    for idx, (batch, target, lang_idx) in enumerate(train_dataloader):
        optim.zero_grad()
        out = model(batch.cuda())
        logits = out[0]
        loss = criterion(logits, target.cuda())
        loss.backward()
        # clip_gradient(model)
        optim.step()
        loss_history.append(loss.item())

        if idx % log_interval == 0:
            interval_loss = np.mean(loss_history[log_interval:])
            print(
                f'{dataset_id} {method_name} Epoch: {epoch}, Step: {idx}, lr={get_lr(optim):.5f}, Training Loss: {interval_loss:.6f}')

    mean_loss = np.mean(interval_loss)
    logfile.add_row(epoch=epoch, measure='tr_loss', value=mean_loss, timelapse=time() - tinit)
    return mean_loss


def test(model, test_dataloader, lang_ids, tinit, epoch, logfile, criterion, measure_prefix):
    print('# Validating model ...')
    loss_history = []
    model.eval()
    langs = lang_ids.keys()
    id_2_lang = {v: k for k, v in lang_ids.items()}
    predictions = {l: [] for l in langs}
    yte_stacked = {l: [] for l in langs}

    for batch, target, lang_idx in test_dataloader:
        out = model(batch.cuda())
        logits = out[0]
        loss = criterion(logits, target.cuda()).item()
        prediction = predict(logits)
        loss_history.append(loss)

        # Assigning prediction to dict in predictions and yte_stacked according to lang_idx
        for i, pred in enumerate(prediction):
            lang_pred = id_2_lang[lang_idx.numpy()[i]]
            predictions[lang_pred].append(pred)
            yte_stacked[lang_pred].append(target[i].detach().cpu().numpy())

    ly = {l: np.vstack(yte_stacked[l]) for l in langs}
    ly_ = {l: np.vstack(predictions[l]) for l in langs}
    l_eval = evaluate(ly, ly_)
    metrics = []
    for lang in langs:
        macrof1, microf1, macrok, microk, macrop, microp, macror, micror = l_eval[lang]
        metrics.append([macrof1, microf1, macrok, microk, macrop, microp, macror, micror])
        if measure_prefix == 'te':
            print(f'Lang {lang}: macro-F1={macrof1:.3f} micro-F1={microf1:.3f}')
    Mf1, mF1, MK, mk, MP, mP, MR, mR = np.mean(np.array(metrics), axis=0)
    print(f'[{measure_prefix}] Averages: MF1, mF1, MP, MR [{Mf1:.5f}, {mF1:.5f}, {MP:.5f}, {MR:.5f}]')

    mean_loss = np.mean(loss_history)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=mF1, timelapse=time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-macro-K', value=MK, timelapse=time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-micro-K', value=mk, timelapse=time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-macro-P', value=MP, timelapse=time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-micro-P', value=mP, timelapse=time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-macro-R', value=MR, timelapse=time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-micro-R', value=mR, timelapse=time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-loss', value=mean_loss, timelapse=time() - tinit)

    return Mf1


def feature_extractor(data, lang_ids, model):
    print('# Feature Extractor Mode...')
    """
    Hidden State = Tuple of torch.FloatTensor (one for the output of the embeddings + one for 
    the output of each layer) of shape (batch_size, sequence_length, hidden_size)
    """
    all_batch_embeddings = {}
    id2lang = {v: k for k, v in lang_ids.items()}
    with torch.no_grad():
        for batch, lang_idx in data:
        # for batch, target, lang_idx in data:
            out = model(batch.cuda())
            last_hidden_state = out[1][-1]
            batch_embeddings = last_hidden_state[:, 0, :]
            for i, l_idx in enumerate(lang_idx.numpy()):
                if id2lang[l_idx] not in all_batch_embeddings.keys():
                    all_batch_embeddings[id2lang[l_idx]] = batch_embeddings[i].detach().cpu().numpy()
                else:
                    all_batch_embeddings[id2lang[l_idx]] = np.vstack((all_batch_embeddings[id2lang[l_idx]],
                                                                      batch_embeddings[i].detach().cpu().numpy()))

    return all_batch_embeddings, id2lang


def new_init_logfile_nn(opt, filename):
    dir_logfile = f'csv_logs/bert/{filename}'
    logfile = CSVLog(dir_logfile, ['dataset', 'method', 'epoch', 'measure', 'value', 'run', 'timelapse'])
    logfile.set_default('dataset', opt.dataset)
    logfile.set_default('run', opt.seed)
    logfile.set_default('method', 'BertGen')
    return logfile


def init_logfile_nn(method_name, opt):
    logfile = CSVLog(opt.logfile_neural, ['dataset', 'method', 'epoch', 'measure', 'value', 'run', 'timelapse'])
    logfile.set_default('dataset', opt.dataset)
    logfile.set_default('run', opt.seed)
    logfile.set_default('method', method_name)
    assert opt.force or not logfile.already_calculated(), f'results for dataset {opt.dataset} method {method_name} ' \
                                                          f'and run {opt.seed} already calculated'
    return logfile