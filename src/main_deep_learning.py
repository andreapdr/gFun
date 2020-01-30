import argparse
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from dataset_builder import MultilingualDataset
from learning.transformers import load_muse_embeddings
from models.lstm_class import RNNMultilingualClassifier
from util.csv_log import CSVLog
from util.early_stop import EarlyStopping
from util.common import *
from util.file import create_if_not_exist
from time import time
from embeddings.pretrained import *
from os.path import join
from tqdm import tqdm
from util.evaluation import evaluate
from util.file import get_file_name

allowed_nets = {'rnn'}

# instantiates the net, initializes the model parameters, and sets embeddings trainable if requested
def init_Net(nC, multilingual_index, xavier_uniform=True):
    net=opt.net
    assert net in allowed_nets, f'{net} not supported, valid ones are={allowed_nets}'

    # instantiate the required net
    if net=='rnn':
        only_post = opt.posteriors and (not opt.pretrained) and (not opt.supervised)
        if only_post:
            print('working on ONLY POST mode')
        model = RNNMultilingualClassifier(
            output_size=nC,
            hidden_size=opt.hidden,
            lvocab_size=multilingual_index.l_vocabsize(),
            learnable_length=opt.learnable,
            lpretrained=multilingual_index.l_embeddings(),
            drop_embedding_range=multilingual_index.sup_range,
            drop_embedding_prop=opt.sup_drop,
            post_probabilities=opt.posteriors,
            only_post=only_post
        )

    # weight initialization
    if xavier_uniform:
        for p in model.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

    if opt.tunable:
        # this has to be performed *after* Xavier initialization is done,
        # otherwise the pretrained embedding parameters will be overrided
        model.finetune_pretrained()

    return model.cuda()


def set_method_name():
    method_name = f'{opt.net}(H{opt.hidden})'
    if opt.pretrained:
        method_name += f'-Muse'
    if opt.supervised:
        method_name += f'-WCE'
    if opt.posteriors:
        method_name += f'-Posteriors'
    if (opt.pretrained or opt.supervised) and opt.tunable:
        method_name+='-(trainable)'
    else:
        method_name += '-(static)'
    if opt.learnable > 0:
        method_name += f'-Learnable{opt.learnable}'
    return method_name


def init_optimizer(model, lr):
    return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=opt.weight_decay)


def init_logfile(method_name, opt):
    logfile = CSVLog(opt.log_file, ['dataset', 'method', 'epoch', 'measure', 'value', 'run', 'timelapse'])
    logfile.set_default('dataset', opt.dataset)
    logfile.set_default('run', opt.seed)
    logfile.set_default('method', method_name)
    assert opt.force or not logfile.already_calculated(), f'results for dataset {opt.dataset} method {method_name} and run {opt.seed} already calculated'
    return logfile


# loads the MUSE embeddings if requested, or returns empty dictionaries otherwise
def load_pretrained_embeddings(we_path, langs):
    lpretrained = lpretrained_vocabulary = none_dict(langs)
    if opt.pretrained:
        lpretrained = load_muse_embeddings(we_path, langs, n_jobs=-1)
        lpretrained_vocabulary = {l: lpretrained[l].vocabulary() for l in langs}
    return lpretrained, lpretrained_vocabulary


# ----------------------------------------------------------------------------------------------------------------------
def main():

    method_name = set_method_name()
    logfile = init_logfile(method_name, opt)

    # Loading the dataset
    data = MultilingualDataset.load(opt.dataset)
    # data.set_view(languages=['de', 'fr', 'sv', 'da', 'es', 'it'])
    data.show_dimensions()
    langs = data.langs()
    l_devel_raw, l_devel_target = data.training(target_as_csr=True)
    l_test_raw, l_test_target = data.test(target_as_csr=True)

    # Loading the MUSE pretrained embeddings (only if requested)
    lpretrained, lpretrained_vocabulary = load_pretrained_embeddings(opt.we_path, langs)

    # Data preparation: indexing / splitting / embedding matrices (pretrained + supervised) / posterior probs
    multilingual_index = MultilingualIndex()
    multilingual_index.index(l_devel_raw, l_devel_target, l_test_raw, lpretrained_vocabulary)
    multilingual_index.train_val_split(val_prop=0.2, max_val=2000, seed=opt.seed)
    multilingual_index.embedding_matrices(lpretrained, opt.supervised)
    if opt.posteriors:
        lPtr, lPva, lPte = multilingual_index.posterior_probabilities(max_training_docs_by_lang=opt.svm_max_docs)
    else:
        lPtr, lPva, lPte = None, None, None

    # Model initialization
    model = init_Net(data.num_categories(), multilingual_index)

    optim = init_optimizer(model, lr=opt.lr)
    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    lr_scheduler = StepLR(optim, step_size=25, gamma=0.5)
    batcher_train = Batch(opt.batch_size, batches_per_epoch=10, languages=langs, lpad=multilingual_index.l_pad())
    batcher_eval  = Batch(opt.batch_size, batches_per_epoch=-1, languages=langs, lpad=multilingual_index.l_pad())

    tinit = time()
    create_if_not_exist(opt.checkpoint_dir)
    early_stop = EarlyStopping(model, patience=opt.patience, checkpoint=f'{opt.checkpoint_dir}/{method_name}-{get_file_name(opt.dataset)}')

    l_train_index, l_train_target = multilingual_index.l_train()
    l_val_index, l_val_target = multilingual_index.l_val()
    l_test_index = multilingual_index.l_test_index()

    print('-'*80)
    print('Start training')
    for epoch in range(1, opt.nepochs + 1):
        train(model, batcher_train, l_train_index, lPtr, l_train_target, tinit, logfile, criterion, optim, epoch, method_name)
        lr_scheduler.step() # reduces the learning rate

        # validation
        macrof1 = test(model, batcher_eval, l_val_index, lPva, l_val_target, tinit, epoch, logfile, criterion, 'va')
        early_stop(macrof1, epoch)
        if opt.test_each>0:
            if (opt.plotmode and (epoch==1 or epoch%opt.test_each==0)) or (not opt.plotmode and epoch%opt.test_each==0 and epoch<opt.nepochs):
                test(model, batcher_eval, l_test_index, lPte, l_test_target, tinit, epoch, logfile, criterion, 'te')

        if early_stop.STOP:
            print('[early-stop] STOP')
            if not opt.plotmode: # with plotmode activated, early-stop is ignored
                break

    # training is over

    # restores the best model according to the Mf1 of the validation set (only when plotmode==False)
    # stoptime = early_stop.stop_time - tinit
    # stopepoch = early_stop.best_epoch
    # logfile.add_row(epoch=stopepoch, measure=f'early-stop', value=early_stop.best_score, timelapse=stoptime)

    if opt.plotmode==False:
        print('-' * 80)
        print('Training over. Performing final evaluation')
        model = early_stop.restore_checkpoint()

        if opt.val_epochs>0:
            print(f'running last {opt.val_epochs} training epochs on the validation set')
            for val_epoch in range(1, opt.val_epochs + 1):
                batcher_train.init_offset()
                train(model, batcher_train, l_val_index, lPva, l_val_target, tinit, logfile, criterion, optim, epoch+val_epoch, method_name)

        # final test
        print('Training complete: testing')
        test(model, batcher_eval, l_test_index, lPte, l_test_target, tinit, epoch, logfile, criterion, 'te')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, batcher, ltrain_index, ltrain_posteriors, lytr, tinit, logfile, criterion, optim, epoch, method_name):
    loss_history = []
    model.train()
    for idx, (batch, post, target, lang) in enumerate(batcher.batchify(ltrain_index, ltrain_posteriors, lytr)):
        optim.zero_grad()
        loss = criterion(model(batch, post, lang), target)
        loss.backward()
        clip_gradient(model)
        optim.step()
        loss_history.append(loss.item())

        if idx % opt.log_interval == 0:
            interval_loss = np.mean(loss_history[-opt.log_interval:])
            print(f'{opt.dataset} {method_name} Epoch: {epoch}, Step: {idx}, lr={get_lr(optim):.5f}, Training Loss: {interval_loss:.6f}')

    mean_loss = np.mean(interval_loss)
    logfile.add_row(epoch=epoch, measure='tr_loss', value=mean_loss, timelapse=time() - tinit)
    return mean_loss


def test(model, batcher, ltest_index, ltest_posteriors, lyte, tinit, epoch, logfile, criterion, measure_prefix):
    model.eval()
    langs = sorted(ltest_index.keys())
    predictions = {l:[] for l in langs}
    yte_stacked = {l:[] for l in langs}
    batcher.init_offset()
    for batch, post, target, lang in tqdm(batcher.batchify(ltest_index, ltest_posteriors, lyte), desc='evaluation: '):
        logits = model(batch, post, lang)
        loss = criterion(logits, target).item()
        prediction = predict(logits)
        predictions[lang].append(prediction)
        yte_stacked[lang].append(target.detach().cpu().numpy())

    ly  = {l:np.vstack(yte_stacked[l]) for l in langs}
    ly_ = {l:np.vstack(predictions[l]) for l in langs}
    l_eval = evaluate(ly, ly_)
    metrics = []
    for lang in langs:
        macrof1, microf1, macrok, microk = l_eval[lang]
        metrics.append([macrof1, microf1, macrok, microk])
        if measure_prefix=='te':
            print(f'Lang {lang}: macro-F1={macrof1:.3f} micro-F1={microf1:.3f}')
        # results.add_row('PolyEmbed_andrea', 'svm', _config_id, config['we_type'],
        #                 (config['max_label_space'], classifier.best_components),
        #                 config['dim_reduction_unsupervised'], op.optimc, op.dataset.split('/')[-1], classifier.time,
        #                 lang, macrof1, microf1, macrok, microk, '')
    Mf1, mF1, MK, mk = np.mean(np.array(metrics), axis=0)
    print(f'[{measure_prefix}] Averages: MF1, mF1, MK, mK [{Mf1:.5f}, {mF1:.5f}, {MK:.5f}, {mk:.5f}]')

    # logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=tend)
    # logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=mf1, timelapse=tend)
    # logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
    # logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-loss', value=loss, timelapse=tend)

    return Mf1


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Neural text classification with Word-Class Embeddings')
    parser.add_argument('dataset', type=str, metavar='datasetpath', help=f'path to the pickled dataset')
    parser.add_argument('--batch-size', type=int, default=50, metavar='int', help='input batch size (default: 100)')
    parser.add_argument('--batch-size-test', type=int, default=250, metavar='int', help='batch size for testing (default: 250)')
    parser.add_argument('--nepochs', type=int, default=200, metavar='int', help='number of epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=10, metavar='int', help='patience for early-stop (default: 10)')
    parser.add_argument('--plotmode', action='store_true', default=False, help='in plot mode executes a long run in order '
                                   'to generate enough data to produce trend plots (test-each should be >0. This mode is '
                                   'used to produce plots, and does not perform an evaluation on the test set.')
    parser.add_argument('--hidden', type=int, default=512, metavar='int', help='hidden lstm size (default: 512)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='float', help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='float', help='weight decay (default: 0)')
    parser.add_argument('--sup-drop', type=float, default=0.5, metavar='[0.0, 1.0]', help='dropout probability for the supervised matrix (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='int', help='random seed (default: 1)')
    parser.add_argument('--svm-max-docs', type=int, default=1000, metavar='int', help='maximum number of documents by '
                              'language used to train the calibrated SVMs (only used if --posteriors is active)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='int', help='how many batches to wait before printing training status')
    parser.add_argument('--log-file', type=str, default='../log/log.csv', metavar='str', help='path to the log csv file')
    # parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str', help=f'if set, specifies the path where to '
    #                     f'save/load the dataset pickled (set to None if you prefer not to retain the pickle file)')
    parser.add_argument('--test-each', type=int, default=0, metavar='int', help='how many epochs to wait before invoking test (default: 0, only at the end)')
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoint', metavar='str', help='path to the directory containing checkpoints')
    parser.add_argument('--net', type=str, default='rnn', metavar='str', help=f'net, one in {allowed_nets}')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use MUSE pretrained embeddings')
    parser.add_argument('--supervised', action='store_true', default=False, help='use supervised embeddings')
    parser.add_argument('--posteriors', action='store_true', default=False, help='concatenate posterior probabilities to doc embeddings')
    parser.add_argument('--learnable', type=int, default=0, metavar='int', help='dimension of the learnable embeddings (default 0)')
    parser.add_argument('--val-epochs', type=int, default=1, metavar='int', help='number of training epochs to perform on the '
                        'validation set once training is over (default 1)')
    parser.add_argument('--we-path', type=str, default='../embeddings', metavar='str',
                        help=f'path to MUSE pretrained embeddings')
    parser.add_argument('--max-label-space', type=int, default=300, metavar='int', help='larger dimension allowed for the '
                        'feature-label embedding (if larger, then PCA with this number of components is applied '
                        '(default 300)')
    parser.add_argument('--force', action='store_true', default=False, help='do not check if this experiment has already been run')
    parser.add_argument('--tunable', action='store_true', default=False,
                        help='pretrained embeddings are tunable from the begining (default False, i.e., static)')

    opt = parser.parse_args()

    assert torch.cuda.is_available(), 'CUDA not available'
    assert not opt.plotmode or opt.test_each > 0, 'plot mode implies --test-each>0'
    # if opt.pickle_dir: opt.pickle_path = join(opt.pickle_dir, f'{opt.dataset}.pickle')
    torch.manual_seed(opt.seed)

    main()
