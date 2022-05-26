import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

N_WORKERS = 8


class RecurrentDataset(Dataset):
    def __init__(self, lX, ly, lPad_index):
        """
        :param lX: dict {lang_id : np.ndarray}
        :param ly:
        """
        self.lX = []
        self.ly = []
        self.lOffset = {}
        self.lPad_index = lPad_index

        for lang, data in lX.items():
            offset = [len(self.lX)]
            self.lX.extend(data)
            offset.append(len(self.lX))
            self.lOffset[lang] = offset

        for lang, target in ly.items():
            self.ly.extend(target)

    def __len__(self):
        return len(self.lX)

    def __getitem__(self, index):
        X = self.lX[index]
        y = self.ly[index]
        return X, y, index, self._get_lang(index)

    def _get_lang(self, index):
        for lang, l_range in self.lOffset.items():
            if index in range(l_range[0], l_range[1]):
                return lang

    def collate_fn(self, data):
        """
        Takes care of padding the batch and also check consistency of batch languages. Groups into dict {lang : lang_batch}
        items sampled from the Dataset class.
        :param data:
        :return:
        """
        lX_batch = {}
        ly_batch = {}
        current_lang = data[0][-1]
        for d in data:
            if d[-1] == current_lang:
                if current_lang not in lX_batch.keys():
                    lX_batch[current_lang] = []
                    ly_batch[current_lang] = []
                lX_batch[current_lang].append(d[0])
                ly_batch[current_lang].append(d[1])
            else:
                current_lang = d[-1]
                lX_batch[current_lang] = []
                ly_batch[current_lang] = []
                lX_batch[current_lang].append(d[0])
                ly_batch[current_lang].append(d[1])

        for lang in lX_batch.keys():
            lX_batch[lang] = self.pad(lX_batch[lang], pad_index=self.lPad_index[lang],
                                      max_pad_length=self.define_pad_length(lX_batch[lang]))
            lX_batch[lang] = torch.LongTensor(lX_batch[lang])
            ly_batch[lang] = torch.FloatTensor(ly_batch[lang])

        return lX_batch, ly_batch

    @staticmethod
    def define_pad_length(index_list):
        lengths = [len(index) for index in index_list]
        return int(np.mean(lengths) + np.std(lengths))

    @staticmethod
    def pad(index_list, pad_index, max_pad_length=None):
        pad_length = np.max([len(index) for index in index_list])
        if max_pad_length is not None:
            pad_length = min(pad_length, max_pad_length)
        for i, indexes in enumerate(index_list):
            index_list[i] = [pad_index] * (pad_length - len(indexes)) + indexes[:pad_length]
        return index_list


class RecurrentDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning Datamodule to be deployed with RecurrentGen.
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    """
    def __init__(self, multilingualIndex, batchsize=64, n_jobs=-1):
        """
        Init RecurrentDataModule.
        :param multilingualIndex: MultilingualIndex, it is a dictionary of training and test documents
        indexed by language code.
        :param batchsize: int, number of sample per batch.
        :param n_jobs: int, number of concurrent workers to be deployed (i.e., parallelizing data loading).
        """
        self.multilingualIndex = multilingualIndex
        self.batchsize = batchsize
        self.n_jobs = n_jobs
        super().__init__()

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            l_train_index, l_train_target = self.multilingualIndex.l_train()
            # Debug settings: reducing number of samples
            # l_train_index = {l: train[:5] for l, train in l_train_index.items()}
            # l_train_target = {l: target[:5] for l, target in l_train_target.items()}

            self.training_dataset = RecurrentDataset(l_train_index, l_train_target,
                                                     lPad_index=self.multilingualIndex.l_pad())

            l_val_index, l_val_target = self.multilingualIndex.l_val()
            # Debug settings: reducing number of samples
            # l_val_index = {l: train[:5] for l, train in l_val_index.items()}
            # l_val_target = {l: target[:5] for l, target in l_val_target.items()}

            self.val_dataset = RecurrentDataset(l_val_index, l_val_target,
                                                lPad_index=self.multilingualIndex.l_pad())
        if stage == 'test' or stage is None:
            l_test_index, l_test_target = self.multilingualIndex.l_test()
            # Debug settings: reducing number of samples
            # l_test_index = {l: train[:5] for l, train in l_test_index.items()}
            # l_test_target = {l: target[:5] for l, target in l_test_target.items()}

            self.test_dataset = RecurrentDataset(l_test_index, l_test_target,
                                                 lPad_index=self.multilingualIndex.l_pad())

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batchsize, num_workers=N_WORKERS,
                          collate_fn=self.training_dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batchsize, num_workers=N_WORKERS,
                          collate_fn=self.val_dataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batchsize, num_workers=N_WORKERS,
                          collate_fn=self.test_dataset.collate_fn)


def tokenize(l_raw, max_len):
    """
    run Bert tokenization on dict {lang: list of samples}.
    :param l_raw:
    :param max_len:
    :return:
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    l_tokenized = {}
    for lang in l_raw.keys():
        output_tokenizer = tokenizer(l_raw[lang], truncation=True, max_length=max_len, padding='max_length')
        l_tokenized[lang] = output_tokenizer['input_ids']
    return l_tokenized


class BertDataModule(RecurrentDataModule):
    """
    Pytorch Lightning Datamodule to be deployed with BertGen.
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    """
    def __init__(self, multilingualIndex, batchsize=64, max_len=512):
        """
        Init BertDataModule.
        :param multilingualIndex: MultilingualIndex, it is a dictionary of training and test documents
        indexed by language code.
        :param batchsize: int, number of sample per batch.
        :param max_len: int, max number of token per document. Absolute cap is 512.
        """
        super().__init__(multilingualIndex, batchsize)
        self.max_len = max_len

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            l_train_raw, l_train_target = self.multilingualIndex.l_train_raw()
            # Debug settings: reducing number of samples
            # print("[NB: DEBUG SETTING RUNNING WITH LOW NUMBER OF SAMPLES]")
            # l_train_raw = {l: train[:100] for l, train in l_train_raw.items()}
            # l_train_target = {l: target[:100] for l, target in l_train_target.items()}

            l_train_index = tokenize(l_train_raw, max_len=self.max_len)
            self.training_dataset = RecurrentDataset(l_train_index, l_train_target,
                                                     lPad_index=self.multilingualIndex.l_pad())

            l_val_raw, l_val_target = self.multilingualIndex.l_val_raw()
            # Debug settings: reducing number of samples
            # print("[NB: DEBUG SETTING RUNNING WITH LOW NUMBER OF SAMPLES]")
            # l_val_raw = {l: train[:50] for l, train in l_val_raw.items()}
            # l_val_target = {l: target[:50] for l, target in l_val_target.items()}

            l_val_index = tokenize(l_val_raw, max_len=self.max_len)
            self.val_dataset = RecurrentDataset(l_val_index, l_val_target,
                                                lPad_index=self.multilingualIndex.l_pad())

        if stage == 'test' or stage is None:
            l_test_raw, l_test_target = self.multilingualIndex.l_test_raw()
            # Debug settings: reducing number of samples
            # l_test_raw = {l: train[:5] for l, train in l_test_raw.items()}
            # l_test_target = {l: target[:5] for l, target in l_test_target.items()}

            l_test_index = tokenize(l_test_raw, max_len=self.max_len)
            self.test_dataset = RecurrentDataset(l_test_index, l_test_target,
                                                 lPad_index=self.multilingualIndex.l_pad())

    def train_dataloader(self):
        """
        NB: Setting n_workers to > 0 will cause "OSError: [Errno 24] Too many open files"
        :return:
        """
        return DataLoader(self.training_dataset, batch_size=self.batchsize)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batchsize)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batchsize)
