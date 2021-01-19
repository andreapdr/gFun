import numpy as np
from scipy.sparse import lil_matrix, issparse
from sklearn.metrics import f1_score, accuracy_score



"""
Scikit learn provides a full set of evaluation metrics, but they treat special cases differently.
I.e., when the number of true positives, false positives, and false negatives ammount to 0, all
affected metrices (precision, recall, and thus f1) output 0 in Scikit learn.
We adhere to the common practice of outputting 1 in this case since the classifier has correctly
classified all examples as negatives.
"""

class ContTable:
    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        self.tp=tp
        self.tn=tn
        self.fp=fp
        self.fn=fn

    def get_d(self): return self.tp + self.tn + self.fp + self.fn

    def get_c(self): return self.tp + self.fn

    def get_not_c(self): return self.tn + self.fp

    def get_f(self): return self.tp + self.fp

    def get_not_f(self): return self.tn + self.fn

    def p_c(self): return (1.0*self.get_c())/self.get_d()

    def p_not_c(self): return 1.0-self.p_c()

    def p_f(self): return (1.0*self.get_f())/self.get_d()

    def p_not_f(self): return 1.0-self.p_f()

    def p_tp(self): return (1.0*self.tp) / self.get_d()

    def p_tn(self): return (1.0*self.tn) / self.get_d()

    def p_fp(self): return (1.0*self.fp) / self.get_d()

    def p_fn(self): return (1.0*self.fn) / self.get_d()

    def tpr(self):
        c = 1.0*self.get_c()
        return self.tp / c if c > 0.0 else 0.0

    def fpr(self):
        _c = 1.0*self.get_not_c()
        return self.fp / _c if _c > 0.0 else 0.0

    def __add__(self, other):
        return ContTable(tp=self.tp + other.tp, tn=self.tn + other.tn, fp=self.fp + other.fp, fn=self.fn + other.fn)

def accuracy(cell):
    return (cell.tp + cell.tn)*1.0 / (cell.tp + cell.fp + cell.fn + cell.tn)

def f1(cell):
    num = 2.0 * cell.tp
    den = 2.0 * cell.tp + cell.fp + cell.fn
    if den>0: return num / den
    #we define f1 to be 1 if den==0 since the classifier has correctly classified all instances as negative
    return 1.0

def K(cell):
    specificity, recall = 0., 0.

    AN = cell.tn + cell.fp
    if AN != 0:
        specificity = cell.tn*1. / AN

    AP = cell.tp + cell.fn
    if AP != 0:
        recall = cell.tp*1. / AP

    if AP == 0:
        return 2. * specificity - 1.
    elif AN == 0:
        return 2. * recall - 1.
    else:
        return specificity + recall - 1.

#computes the (hard) counters tp, fp, fn, and tn fron a true and predicted vectors of hard decisions
#true_labels and predicted_labels are two vectors of shape (number_documents,)
def hard_single_metric_statistics(true_labels, predicted_labels):
    assert len(true_labels)==len(predicted_labels), "Format not consistent between true and predicted labels."
    nd = len(true_labels)
    tp = np.sum(predicted_labels[true_labels==1])
    fp = np.sum(predicted_labels[true_labels == 0])
    fn = np.sum(true_labels[predicted_labels == 0])
    tn = nd - (tp+fp+fn)
    return ContTable(tp=tp, tn=tn, fp=fp, fn=fn)

#computes the (soft) contingency table where tp, fp, fn, and tn are the cumulative masses for the posterioir
# probabilitiesfron with respect to the true binary labels
#true_labels and posterior_probabilities are two vectors of shape (number_documents,)
def soft_single_metric_statistics(true_labels, posterior_probabilities):
    assert len(true_labels)==len(posterior_probabilities), "Format not consistent between true and predicted labels."
    tp = np.sum(posterior_probabilities[true_labels == 1])
    fn = np.sum(1. - posterior_probabilities[true_labels == 1])
    fp = np.sum(posterior_probabilities[true_labels == 0])
    tn = np.sum(1. - posterior_probabilities[true_labels == 0])
    return ContTable(tp=tp, tn=tn, fp=fp, fn=fn)

#if the classifier is single class, then the prediction is a vector of shape=(nD,) which causes issues when compared
#to the true labels (of shape=(nD,1)). This method increases the dimensions of the predictions.
def __check_consistency_and_adapt(true_labels, predictions):
    if predictions.ndim == 1:
        return __check_consistency_and_adapt(true_labels, np.expand_dims(predictions, axis=1))
    if true_labels.ndim == 1:
        return __check_consistency_and_adapt(np.expand_dims(true_labels, axis=1),predictions)
    if true_labels.shape != predictions.shape:
        raise ValueError("True and predicted label matrices shapes are inconsistent %s %s."
                         % (true_labels.shape, predictions.shape))
    _,nC = true_labels.shape
    return true_labels, predictions, nC

def macro_average(true_labels, predicted_labels, metric, metric_statistics=hard_single_metric_statistics):
    true_labels, predicted_labels, nC = __check_consistency_and_adapt(true_labels, predicted_labels)
    return np.mean([metric(metric_statistics(true_labels[:, c], predicted_labels[:, c])) for c in range(nC)])

def micro_average(true_labels, predicted_labels, metric, metric_statistics=hard_single_metric_statistics):
    true_labels, predicted_labels, nC = __check_consistency_and_adapt(true_labels, predicted_labels)

    accum = ContTable()
    for c in range(nC):
        other = metric_statistics(true_labels[:, c], predicted_labels[:, c])
        accum = accum + other

    return metric(accum)

#true_labels and predicted_labels are two matrices in sklearn.preprocessing.MultiLabelBinarizer format
def macroF1(true_labels, predicted_labels):
    return macro_average(true_labels,predicted_labels, f1)

#true_labels and predicted_labels are two matrices in sklearn.preprocessing.MultiLabelBinarizer format
def microF1(true_labels, predicted_labels):
    return micro_average(true_labels, predicted_labels, f1)

#true_labels and predicted_labels are two matrices in sklearn.preprocessing.MultiLabelBinarizer format
def macroK(true_labels, predicted_labels):
    return macro_average(true_labels,predicted_labels, K)

#true_labels and predicted_labels are two matrices in sklearn.preprocessing.MultiLabelBinarizer format
def microK(true_labels, predicted_labels):
    return micro_average(true_labels, predicted_labels, K)

#true_labels is a matrix in sklearn.preprocessing.MultiLabelBinarizer format and posterior_probabilities is a matrix
#of the same shape containing real values in [0,1]
def smoothmacroF1(true_labels, posterior_probabilities):
    return macro_average(true_labels,posterior_probabilities, f1, metric_statistics=soft_single_metric_statistics)

#true_labels is a matrix in sklearn.preprocessing.MultiLabelBinarizer format and posterior_probabilities is a matrix
#of the same shape containing real values in [0,1]
def smoothmicroF1(true_labels, posterior_probabilities):
    return micro_average(true_labels, posterior_probabilities, f1, metric_statistics=soft_single_metric_statistics)

#true_labels is a matrix in sklearn.preprocessing.MultiLabelBinarizer format and posterior_probabilities is a matrix
#of the same shape containing real values in [0,1]
def smoothmacroK(true_labels, posterior_probabilities):
    return macro_average(true_labels,posterior_probabilities, K, metric_statistics=soft_single_metric_statistics)

#true_labels is a matrix in sklearn.preprocessing.MultiLabelBinarizer format and posterior_probabilities is a matrix
#of the same shape containing real values in [0,1]
def smoothmicroK(true_labels, posterior_probabilities):
    return micro_average(true_labels, posterior_probabilities, K, metric_statistics=soft_single_metric_statistics)




"""
Scikit learn provides a full set of evaluation metrics, but they treat special cases differently.
I.e., when the number of true positives, false positives, and false negatives ammount to 0, all
affected metrices (precision, recall, and thus f1) output 0 in Scikit learn.
We adhere to the common practice of outputting 1 in this case since the classifier has correctly
classified all examples as negatives.
"""

def evaluation(y_true, y_pred, classification_type):

    if classification_type == 'multilabel':
        eval_function = multilabel_eval
    elif classification_type == 'singlelabel':
        eval_function = singlelabel_eval

    Mf1, mf1, accuracy = eval_function(y_true, y_pred)

    return Mf1, mf1, accuracy


def multilabel_eval(y, y_):

    tp = y.multiply(y_)

    fn = lil_matrix(y.shape)
    true_ones = y==1
    fn[true_ones]=1-tp[true_ones]

    fp = lil_matrix(y.shape)
    pred_ones = y_==1
    if pred_ones.nnz>0:
        fp[pred_ones]=1-tp[pred_ones]

    #macro-f1
    tp_macro = np.asarray(tp.sum(axis=0), dtype=int).flatten()
    fn_macro = np.asarray(fn.sum(axis=0), dtype=int).flatten()
    fp_macro = np.asarray(fp.sum(axis=0), dtype=int).flatten()

    pos_pred = tp_macro+fp_macro
    pos_true = tp_macro+fn_macro
    prec=np.zeros(shape=tp_macro.shape,dtype=float)
    rec=np.zeros(shape=tp_macro.shape,dtype=float)
    np.divide(tp_macro, pos_pred, out=prec, where=pos_pred>0)
    np.divide(tp_macro, pos_true, out=rec, where=pos_true>0)
    den=prec+rec

    macrof1=np.zeros(shape=tp_macro.shape,dtype=float)
    np.divide(np.multiply(prec,rec),den,out=macrof1,where=den>0)
    macrof1 *=2

    macrof1[(pos_pred==0)*(pos_true==0)]=1
    macrof1 = np.mean(macrof1)

    #micro-f1
    tp_micro = tp_macro.sum()
    fn_micro = fn_macro.sum()
    fp_micro = fp_macro.sum()
    pos_pred = tp_micro + fp_micro
    pos_true = tp_micro + fn_micro
    prec = (tp_micro / pos_pred) if pos_pred>0 else 0
    rec  = (tp_micro / pos_true) if pos_true>0 else 0
    den = prec+rec
    microf1 = 2*prec*rec/den if den>0 else 0
    if pos_pred==pos_true==0:
        microf1=1

    #accuracy
    ndecisions = np.multiply(*y.shape)
    tn = ndecisions - (tp_micro+fn_micro+fp_micro)
    acc = (tp_micro+tn)/ndecisions

    return macrof1,microf1,acc


def singlelabel_eval(y, y_):
    if issparse(y_): y_ = y_.toarray().flatten()
    macrof1 = f1_score(y, y_, average='macro')
    microf1 = f1_score(y, y_, average='micro')
    acc = accuracy_score(y, y_)
    return macrof1,microf1,acc

