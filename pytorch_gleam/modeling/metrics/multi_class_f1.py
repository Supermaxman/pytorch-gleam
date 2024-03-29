from typing import List, Optional

from sklearn.metrics import precision_recall_fscore_support

from pytorch_gleam.modeling.metrics.base_metrics import Metric


class F1PRMultiClassMetric(Metric):
    def __init__(self, num_classes: int, mode: str = "macro", pos_labels: Optional[List[int]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.mode = mode
        if pos_labels is None:
            pos_labels = list(range(self.num_classes))[1:]

        self.pos_labels = pos_labels

    def forward(self, labels, predictions):
        # labels = labels.numpy()
        # predictions = predictions.numpy()
        cls_precision, cls_recall, cls_f1, cls_sup = precision_recall_fscore_support(
            labels, predictions, average=None, labels=self.pos_labels, zero_division=0
        )
        if self.mode == "macro":
            # ignore 0 class in macro average
            mode_f1 = cls_f1.mean()
            mode_precision = cls_precision.mean()
            mode_recall = cls_recall.mean()
        elif self.mode == "micro":
            (mode_precision, mode_recall, mode_f1, mode_sup,) = precision_recall_fscore_support(
                labels,
                predictions,
                average="micro",
                labels=self.pos_labels,
                zero_division=0,
            )
        else:
            raise ValueError(f"Unknown metric mode: {self.mode}")

        return (
            mode_f1,
            mode_precision,
            mode_recall,
            cls_f1,
            cls_precision,
            cls_recall,
            self.pos_labels,
        )
