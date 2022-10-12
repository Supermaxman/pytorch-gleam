from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, pos_scores, neg_scores, normalizer=None):
        pass

    def calculate_scores(self, pos_scores, neg_scores=None):
        if neg_scores is None:
            return pos_scores
        scores = torch.cat([pos_scores, neg_scores], dim=0)
        return scores


class MarginContrastiveLoss(ContrastiveLoss):
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores, neg_scores, normalizer=None):
        margin = self.margin
        if normalizer is not None:
            margin = self.margin / normalizer
        loss = torch.relu(pos_scores - neg_scores + margin)
        return loss

    def calculate_scores(self, pos_scores, neg_scores=None):
        scores = super().calculate_scores(pos_scores, neg_scores)
        return -scores


class MarginSigmoidContrastiveLoss(MarginContrastiveLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pos_scores, neg_scores, normalizer=None):
        pos_scores = torch.sigmoid(pos_scores)
        neg_scores = torch.sigmoid(neg_scores)
        loss = torch.relu(pos_scores - neg_scores + self.margin)
        return loss

    def calculate_scores(self, pos_scores, neg_scores=None):
        scores = super().calculate_scores(pos_scores, neg_scores)
        scores = 1.0 - torch.sigmoid(scores)
        return scores


class ProbContrastiveLoss(ContrastiveLoss):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pos_scores, neg_scores, normalizer=None):
        pos_bce = self.criterion(pos_scores, torch.ones_like(pos_scores))
        neg_bce = self.criterion(neg_scores, torch.zeros_like(neg_scores))

        # minimize positive bce while maximizing negative bce
        loss = pos_bce + neg_bce

        return loss
