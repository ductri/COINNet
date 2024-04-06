from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import unified_metrics



class MyClusterAccuracy(Metric):
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self) -> Tensor:
        # parse inputs
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        acc = Tensor([unified_metrics.best_acc(targets, preds, self.K)])

        return acc
