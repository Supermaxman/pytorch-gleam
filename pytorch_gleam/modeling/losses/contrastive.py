from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, pos_scores, neg_scores):
        pass


class MarginContrastiveLoss(ContrastiveLoss):
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores, neg_scores):
        # [bsize, 1]
        # [bsize, 1]
        # TODO double check margin loss
        loss = torch.relu(pos_scores - neg_scores + self.margin)
        return loss
