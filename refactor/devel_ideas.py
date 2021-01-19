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
