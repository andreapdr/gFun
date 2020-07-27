"""
Test with smaller subset of languages.

1. Load doc (RCV1/2)
2. Tokenize texts via bertTokenizer (I should already have these dumps)
3. Construct better Dataloader/Datasets. NB: I need to keep track of the languages only for
the testing phase (but who cares actually? If I have to do it for the testing phase, I think
it is better to deploy it also in the training phase...)
4. ...
5. I have to understand if the pooled hidden state of the last layer is way worse than its averaged
version (However, in BertForSeqClassification I guess that the pooled version is passed through
the output linear layer in order to get the prediction scores?)
6. At the same time, I have to build also an end-to-end model in order to fine-tune it. The previous step
would be useful when deploying mBert as a View Generator. (Refactor gFun code with view generators?)
7. ...
8. Profits

"""
from dataset_builder import MultilingualDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from util.common import clip_gradient, predict
from time import time
from util.csv_log import CSVLog
from util.evaluation import evaluate
from util.early_stop import EarlyStopping
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
import argparse


def get_model(n_out):
    print('# Initializing model ...')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=n_out)
    return model

def set_method_name():
    return 'mBERT'

def init_optimizer(model, lr):
    # return AdamW(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opt.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': opt.weight_decay}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer

def init_logfile(method_name, opt):
    logfile = CSVLog(opt.log_file, ['dataset', 'method', 'epoch', 'measure', 'value', 'run', 'timelapse'])
    logfile.set_default('dataset', opt.dataset)
    logfile.set_default('run', opt.seed)
    logfile.set_default('method', method_name)
    assert opt.force or not logfile.already_calculated(), f'results for dataset {opt.dataset} method {method_name} and run {opt.seed} already calculated'
    return logfile

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_dataset_name(datapath):
    possible_splits = [str(i) for i in range(10)]
    splitted = datapath.split('_')
    id_split = splitted[-1].split('.')[0][-1]
    if id_split in possible_splits:
        dataset_name = splitted[0].split('/')[-1]
        return f'{dataset_name}_run{id_split}'

def load_datasets(datapath):
    data = MultilingualDataset.load(datapath)
    data.set_view(languages=['nl'])   # Testing with just two langs
    data.show_dimensions()

    l_devel_raw, l_devel_target = data.training(target_as_csr=False)
    l_test_raw, l_test_target = data.test(target_as_csr=False)

    return l_devel_raw, l_devel_target, l_test_raw, l_test_target


def do_tokenization(l_dataset, max_len=512):
    print('# Starting Tokenization ...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    langs = l_dataset.keys()
    l_tokenized = {}
    for lang in langs:
        l_tokenized[lang] = tokenizer(l_dataset[lang],
                                      truncation=True,
                                      max_length=max_len,
                                      add_special_tokens=True,
                                      padding='max_length')
    return l_tokenized


class TrainingDataset(Dataset):
    """
    data: dict of lang specific tokenized data
    labels: dict of lang specific targets
    """
    def __init__(self, data, labels):
        self.langs = data.keys()
        self.lang_ids = {lang:identifier for identifier, lang in enumerate(self.langs)}

        for i, lang in enumerate(self.langs):
            # print(lang)
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
        # return x, y, lang

    def get_lang_ids(self):
        return self.lang_ids

def freeze_encoder(model):
    for param in model.base_model.parameters():
        param.requires_grad = False
    return model

def check_param_grad_status(model):
    print('#'*50)
    print('Model paramater status')
    for name, child in model.named_children():
        trainable = False
        for param in child.parameters():
            if param.requires_grad:
                trainable = True
        if not trainable:
            print(f'{name} is frozen')
        else:
            print(f'{name} is not frozen')
    print('#'*50)

def train(model, train_dataloader, epoch, criterion, optim, method_name, tinit, logfile):
    _dataset_path = opt.dataset.split('/')[-1].split('_')
    # dataset_id = 'RCV1/2_run0_newBert'
    dataset_id = _dataset_path[0] + _dataset_path[-1]

    loss_history = []
    model.train()

    for idx, (batch, target, lang_idx) in enumerate(train_dataloader):
        # optim.zero_grad()
        out = model(batch.cuda())
        loss = criterion(out[0], target.cuda())
        loss.backward()
        clip_gradient(model)
        optim.step()
        loss_history.append(loss.item())

        if idx % opt.log_interval == 0:
            interval_loss = np.mean(loss_history[-opt.log_interval:])
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
    id_2_lang = {v:k for k,v in lang_ids.items()}
    predictions = {l: [] for l in langs}
    yte_stacked = {l: [] for l in langs}

    for batch, target, lang_idx in test_dataloader:
        out = model(batch.cuda())
        logits = out[0]
        loss = criterion(logits, target.cuda()).item()
        prediction = predict(logits)
        loss_history.append(loss)

        # Assigning prediction to dict in predictionS and yte_stacked according to lang_idx
        for i, pred in enumerate(prediction):
            lang_pred = id_2_lang[lang_idx.numpy()[i]]
            predictions[lang_pred].append(pred)
            yte_stacked[lang_pred].append(target[i].detach().cpu().numpy())

    ly = {l: np.vstack(yte_stacked[l]) for l in langs}
    ly_ = {l: np.vstack(predictions[l]) for l in langs}
    l_eval = evaluate(ly, ly_)
    metrics = []
    for lang in langs:
        macrof1, microf1, macrok, microk = l_eval[lang]
        metrics.append([macrof1, microf1, macrok, microk])
        if measure_prefix == 'te':
            print(f'Lang {lang}: macro-F1={macrof1:.3f} micro-F1={microf1:.3f}')
    Mf1, mF1, MK, mk = np.mean(np.array(metrics), axis=0)
    print(f'[{measure_prefix}] Averages: MF1, mF1, MK, mK [{Mf1:.5f}, {mF1:.5f}, {MK:.5f}, {mk:.5f}]')

    mean_loss = np.mean(loss_history)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=mF1, timelapse=time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-macro-K', value=MK, timelapse=time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-micro-K', value=mk, timelapse=time() - tinit)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-loss', value=mean_loss, timelapse=time() - tinit)

    return Mf1

def get_tr_val_split(l_tokenized_tr, l_devel_target, val_prop, max_val, seed):
    l_split_va = l_tokenized_tr
    l_split_val_target = {l: [] for l in l_tokenized_tr.keys()}
    l_split_tr = l_tokenized_tr
    l_split_tr_target = {l: [] for l in l_tokenized_tr.keys()}

    for lang in l_tokenized_tr.keys():
        val_size = int(min(len(l_tokenized_tr[lang]['input_ids']) * val_prop, max_val))

        l_split_tr[lang]['input_ids'], l_split_va[lang]['input_ids'], l_split_tr_target[lang], l_split_val_target[lang] = \
            train_test_split(l_tokenized_tr[lang]['input_ids'], l_devel_target[lang], test_size=val_size, random_state=seed, shuffle=True)

    return  l_split_tr, l_split_tr_target, l_split_va, l_split_val_target

def main():
    print('Running main ...')

    DATAPATH = opt.dataset
    method_name = set_method_name()
    logfile = init_logfile(method_name, opt)

    l_devel_raw, l_devel_target, l_test_raw, l_test_target = load_datasets(DATAPATH)
    l_tokenized_tr = do_tokenization(l_devel_raw, max_len=512)

    l_split_tr, l_split_tr_target, l_split_va, l_split_val_target = get_tr_val_split(l_tokenized_tr, l_devel_target, val_prop=0.2, max_val=2000, seed=opt.seed)

    l_tokenized_te = do_tokenization(l_test_raw, max_len=512)

    tr_dataset = TrainingDataset(l_split_tr, l_split_tr_target)
    va_dataset = TrainingDataset(l_split_va, l_split_val_target)
    te_dataset = TrainingDataset(l_tokenized_te, l_test_target)

    tr_dataloader = DataLoader(tr_dataset, batch_size=4, shuffle=True)
    va_dataloader = DataLoader(va_dataset, batch_size=2, shuffle=False)
    te_dataloader = DataLoader(te_dataset, batch_size=2, shuffle=False)

    # Initializing model
    model = get_model(73)
    model = model.cuda()
    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    optim = init_optimizer(model, lr=opt.lr)
    # lr_scheduler = StepLR(optim, step_size=25, gamma=0.5)
    early_stop = EarlyStopping(model, optimizer=optim, patience=opt.patience,
                               checkpoint=f'{opt.checkpoint_dir}/{method_name}-{get_dataset_name(opt.dataset)}')
    # lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optim, num_warmup_steps= , num_training_steps=)
    # print(model)

    # Freezing encoder
    # model = freeze_encoder(model)
    check_param_grad_status(model)

    # Training loop
    tinit = time()
    lang_ids = va_dataset.lang_ids
    for epoch in range(1, opt.nepochs+1):
        print('# Start Training ...')
        train(model, tr_dataloader, epoch, criterion, optim, 'TestingBert', tinit, logfile)
        # lr_scheduler.step(epoch=None) # reduces the learning rate

        # validation
        macrof1 = test(model, va_dataloader, lang_ids, tinit, epoch, logfile, criterion, 'va')
        early_stop(macrof1, epoch)
        if opt.test_each>0:
            if (opt.plotmode and (epoch==1 or epoch%opt.test_each==0)) or (not opt.plotmode and epoch%opt.test_each==0 and epoch<opt.nepochs):
                test(model, te_dataloader, lang_ids, tinit, epoch, logfile, criterion, 'te')

        if early_stop.STOP:
            print('[early-stop] STOP')
            if not opt.plotmode:
                break

    if opt.plotmode==False:
        print('-' * 80)
        print('Training over. Performing final evaluation')

        model = early_stop.restore_checkpoint()

        if opt.val_epochs>0:
            print(f'running last {opt.val_epochs} training epochs on the validation set')
            for val_epoch in range(1, opt.val_epochs + 1):
                train(model, va_dataloader, epoch+val_epoch, criterion, optim, 'TestingBert', tinit, logfile)

        # final test
        print('Training complete: testing')
        test(model, te_dataloader, lang_ids, tinit, epoch, logfile, criterion, 'te')

    exit('Code Executed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural text classification with Word-Class Embeddings - mBert model')

    parser.add_argument('--dataset', type=str, default='/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle',
                        metavar='datasetpath', help=f'path to the pickled dataset')
    parser.add_argument('--nepochs', type=int, default=200, metavar='int',
                        help='number of epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=2e-5, metavar='float',
                        help='learning rate (default: 2e-5)')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='float',
                        help='weight decay (default: 0)')
    parser.add_argument('--patience', type=int, default=10, metavar='int',
                        help='patience for early-stop (default: 10)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='int',
                        help='how many batches to wait before printing training status')
    parser.add_argument('--log-file', type=str, default='../log/log_mBert.csv', metavar='str',
                        help='path to the log csv file')
    parser.add_argument('--seed', type=int, default=1, metavar='int', help='random seed (default: 1)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='do not check if this experiment has already been run')
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoint', metavar='str',
                        help='path to the directory containing checkpoints')
    parser.add_argument('--plotmode', action='store_true', default=False,
                        help='in plot mode executes a long run in order '
                             'to generate enough data to produce trend plots (test-each should be >0. This mode is '
                             'used to produce plots, and does not perform an evaluation on the test set.')
    parser.add_argument('--test-each', type=int, default=0, metavar='int',
                        help='how many epochs to wait before invoking test (default: 0, only at the end)')
    parser.add_argument('--val-epochs', type=int, default=1, metavar='int',
                        help='number of training epochs to perform on the validation set once training is over (default 1)')
    opt = parser.parse_args()

    # Testing different parameters ...
    opt.weight_decay = 0.01
    opt.patience = 5

    main()
    # TODO: refactor .cuda() -> .to(device) in order to check if the process is faster on CPU given the bigger batch size 