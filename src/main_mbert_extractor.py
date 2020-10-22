from main_mbert import *
import pickle


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


def feature_extractor(data, lang_ids, model_path='/home/andreapdr/funneling_pdr/hug_checkpoint/mBERT-jrc_run0/'):
    print('# Feature Extractor Mode...')
    from transformers import BertConfig
    config = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True, num_labels=300)
    model = BertForSequenceClassification.from_pretrained(model_path,
                                                          config=config).cuda()

    """
    Hidden State = Tuple of torch.FloatTensor (one for the output of the embeddings + one for 
    the output of each layer) of shape (batch_size, sequence_length, hidden_size)
    """
    all_batch_embeddings = {}
    id2lang = {v:k for k,v in lang_ids.items()}
    with torch.no_grad():
        for batch, target, lang_idx in data:
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


def main():
    print('Running main ...')
    print(f'Model path: {opt.modelpath}\nDataset path: {opt.dataset}')
    DATAPATH = opt.dataset
    MAX_LEN = 512

    l_devel_raw, l_devel_target, l_test_raw, l_test_target = load_datasets(DATAPATH)
    l_tokenized_tr = do_tokenization(l_devel_raw, max_len=MAX_LEN)
    l_tokenized_te = do_tokenization(l_test_raw, max_len=MAX_LEN)

    tr_dataset = TrainingDataset(l_tokenized_tr, l_devel_target)
    tr_lang_ids = tr_dataset.lang_ids

    te_dataset = TrainingDataset(l_tokenized_te, l_test_target)
    te_lang_ids = te_dataset.lang_ids

    tr_dataloader = DataLoader(tr_dataset, batch_size=64, shuffle=False)  # Shuffle False to extract doc embeddings
    te_dataloader = DataLoader(te_dataset, batch_size=64, shuffle=False)  # Shuffle False to extract doc

    tr_all_batch_embeddings, id2lang_tr = feature_extractor(tr_dataloader, tr_lang_ids, opt.modelpath)    # Extracting doc embed for devel
    with open(f'{opt.modelpath}/TR_embed_{get_dataset_name(opt.dataset)}.pkl', 'wb') as outfile:
        pickle.dump((tr_all_batch_embeddings, id2lang_tr), outfile)

    te_all_batch_embeddings, id2lang_te = feature_extractor(te_dataloader, te_lang_ids, opt.modelpath)    # Extracting doc embed for test
    with open(f'{opt.modelpath}/TE_embed_{get_dataset_name(opt.dataset)}.pkl', 'wb') as outfile:
        pickle.dump((te_all_batch_embeddings, id2lang_te), outfile)

    exit('Extraction completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mBert model document embedding extractor')

    parser.add_argument('--dataset', type=str,
                        default='/home/moreo/CLESA/rcv2/rcv1-2_doclist_trByLang1000_teByLang1000_processed_run0.pickle',
                        metavar='datasetpath', help=f'path to the pickled dataset')
    parser.add_argument('--seed', type=int, default=1, metavar='int', help='random seed (default: 1)')
    parser.add_argument('--modelpath', type=str, default='/home/andreapdr/funneling_pdr/hug_checkpoint/mBERT-jrc_run0',
                        metavar='modelpath', help=f'path to pre-trained mBert model')
    opt = parser.parse_args()

    main()

