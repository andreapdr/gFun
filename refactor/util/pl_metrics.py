import torch
from pytorch_lightning.metrics import Metric
from util.common import is_false, is_true


class CustomF1(Metric):
    def __init__(self, num_classes, device, average='micro'):
        """
        Custom F1 metric.
        Scikit learn provides a full set of evaluation metrics, but they treat special cases differently.
        I.e., when the number of true positives, false positives, and false negatives amount to 0, all
        affected metrics (precision, recall, and thus f1) output 0 in Scikit learn.
        We adhere to the common practice of outputting 1 in this case since the classifier has correctly
        classified all examples as negatives.
        :param num_classes:
        :param device:
        :param average:
        """
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.device = 'cuda' if device else 'cpu'
        self.add_state('true_positive', default=torch.zeros(self.num_classes))
        self.add_state('true_negative', default=torch.zeros(self.num_classes))
        self.add_state('false_positive', default=torch.zeros(self.num_classes))
        self.add_state('false_negative', default=torch.zeros(self.num_classes))

    def update(self, preds, target):
        true_positive, true_negative, false_positive, false_negative = self._update(preds, target)

        self.true_positive += true_positive
        self.true_negative += true_negative
        self.false_positive += false_positive
        self.false_negative += false_negative

    def _update(self, pred, target):
        assert pred.shape == target.shape
        # preparing preds and targets for count
        true_pred = is_true(pred, self.device)
        false_pred = is_false(pred, self.device)
        true_target = is_true(target, self.device)
        false_target = is_false(target, self.device)

        tp = torch.sum(true_pred * true_target, dim=0)
        tn = torch.sum(false_pred * false_target, dim=0)
        fp = torch.sum(true_pred * false_target, dim=0)
        fn = torch.sum(false_pred * target, dim=0)
        return tp, tn, fp, fn

    def compute(self):
        if self.average == 'micro':
            num = 2.0 * self.true_positive.sum()
            den = 2.0 * self.true_positive.sum() + self.false_positive.sum() + self.false_negative.sum()
            if den > 0:
                return (num / den).to(self.device)
            return torch.FloatTensor([1.]).to(self.device)
        if self.average == 'macro':
            class_specific = []
            for i in range(self.num_classes):
                class_tp = self.true_positive[i]
                class_tn = self.true_negative[i]
                class_fp = self.false_positive[i]
                class_fn = self.false_negative[i]
                num = 2.0 * class_tp
                den = 2.0 * class_tp + class_fp + class_fn
                if den > 0:
                    class_specific.append(num / den)
                else:
                    class_specific.append(1.)
            average = torch.sum(torch.Tensor(class_specific))/self.num_classes
            return average.to(self.device)
