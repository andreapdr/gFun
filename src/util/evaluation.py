from joblib import Parallel, delayed

from src.util.metrics import *


def evaluation_metrics(y, y_):
    if len(y.shape) == len(y_.shape) == 1 and len(np.unique(y)) > 2:  # single-label
        raise NotImplementedError()  # return f1_score(y,y_,average='macro'), f1_score(y,y_,average='micro')
    else:  # the metrics I implemented assume multiclass multilabel classification as binary classifiers
        return macroF1(y, y_), microF1(y, y_), macroK(y, y_), microK(y, y_), macroP(y, y_), microP(y, y_), macroR(y, y_), microR(y, y_)


def evaluate(ly_true, ly_pred, metrics=evaluation_metrics, n_jobs=-1):
    if n_jobs == 1:
        return {lang: metrics(ly_true[lang], ly_pred[lang]) for lang in ly_true.keys()}
    else:
        langs = list(ly_true.keys())
        evals = Parallel(n_jobs=n_jobs)(delayed(metrics)(ly_true[lang], ly_pred[lang]) for lang in langs)
        return {lang: evals[i] for i, lang in enumerate(langs)}
