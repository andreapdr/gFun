# from sklearn.externals.joblib import Parallel, delayed
from joblib import Parallel, delayed
from util.metrics import *
from sklearn.metrics import f1_score
import numpy as np
import time


def evaluation_metrics(y, y_):
    if len(y.shape)==len(y_.shape)==1 and len(np.unique(y))>2: #single-label
        raise NotImplementedError()#return f1_score(y,y_,average='macro'), f1_score(y,y_,average='micro')
    else: #the metrics I implemented assume multiclass multilabel classification as binary classifiers
        return macroF1(y, y_), microF1(y, y_), macroK(y, y_), microK(y, y_)


def soft_evaluation_metrics(y, y_):
    if len(y.shape)==len(y_.shape)==1 and len(np.unique(y))>2: #single-label
        raise NotImplementedError()#return f1_score(y,y_,average='macro'), f1_score(y,y_,average='micro')
    else: #the metrics I implemented assume multiclass multilabel classification as binary classifiers
        return smoothmacroF1(y, y_), smoothmicroF1(y, y_), smoothmacroK(y, y_), smoothmicroK(y, y_)


def evaluate(ly_true, ly_pred, metrics=evaluation_metrics, n_jobs=-1):
    print('evaluation (n_jobs={})'.format(n_jobs))
    if n_jobs == 1:
        return {lang: metrics(ly_true[lang], ly_pred[lang]) for lang in ly_true.keys()}
    else:
        langs = list(ly_true.keys())
        evals = Parallel(n_jobs=n_jobs)(delayed(metrics)(ly_true[lang], ly_pred[lang]) for lang in langs)
        return {lang: evals[i] for i, lang in enumerate(langs)}


def average_results(l_eval, show=True):
    metrics  = []
    for lang in l_eval.keys():
        macrof1, microf1, macrok, microk = l_eval[lang]
        metrics.append([macrof1, microf1, macrok, microk])
        if show:
            print('Lang %s: macro-F1=%.3f micro-F1=%.3f' % (lang, macrof1, microf1))

    ave = np.mean(np.array(metrics), axis=0)
    if show:
        print('Averages: MF1, mF1, MK, mK', ave)
    return ave


def evaluate_method(polylingual_method, lX, ly, predictor=None, soft=False, return_time=False):
    tinit=time.time()
    print('prediction for test')
    assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
    n_jobs = polylingual_method.n_jobs if hasattr(polylingual_method, 'n_jobs') else -1

    if predictor is None:
        predictor = polylingual_method.predict

    metrics = evaluation_metrics
    if soft is True:
        metrics = soft_evaluation_metrics
    ly_ = predictor(lX, ly)

    eval_ = evaluate(ly, ly_, metrics=metrics, n_jobs=n_jobs)
    if return_time:
        return eval_, time.time()-tinit
    else:
        return eval_


def evaluate_single_lang(polylingual_method, X, y, lang, predictor=None, soft=False):
    print('prediction for test in a single language')
    if predictor is None:
        predictor = polylingual_method.predict

    metrics = evaluation_metrics
    if soft is True:
        metrics = soft_evaluation_metrics

    ly_ = predictor({lang:X})
    return metrics(y, ly_[lang])


def get_binary_counters(polylingual_method, lX, ly, predictor=None):
    print('prediction for test')
    assert set(lX.keys()) == set(ly.keys()), 'inconsistent dictionaries in evaluate'
    n_jobs = polylingual_method.n_jobs
    if predictor is None:
        predictor = polylingual_method.predict
    ly_ = predictor(lX)
    print('evaluation (n_jobs={})'.format(n_jobs))
    if n_jobs == 1:
        return {lang: binary_counters(ly[lang], ly_[lang]) for lang in ly.keys()}
    else:
        langs = list(ly.keys())
        evals = Parallel(n_jobs=n_jobs)(delayed(binary_counters)(ly[lang], ly_[lang]) for lang in langs)
        return {lang: evals[i] for i, lang in enumerate(langs)}


def binary_counters(y, y_):
    y = np.reshape(y, (-1))
    assert y.shape==y_.shape and len(y.shape)==1, 'error, binary vector expected'
    counters = hard_single_metric_statistics(y, y_)
    return counters.tp, counters.tn, counters.fp, counters.fn

