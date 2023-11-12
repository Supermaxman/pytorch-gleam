from typing import Dict
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ValuesModule(nn.module, ABC):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        values_map: Dict[str, int],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.values_map = values_map
        self.num_values = len(values_map)
        self.value_embeddings = nn.Embedding(self.num_values, self.output_dim)

    @abstractmethod
    def forward(self, embeddings, embeddings_mask, value_mask, post_mask=None, frame_mask=None):
        pass


class ValueAttentionPooling(ValuesModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.input_dim != self.output_dim:
            self.linear = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.linear = None
        self.value_softmax = nn.Softmax(dim=-1)

    def forward(self, embeddings, embeddings_mask, value_mask, post_mask=None, frame_mask=None):
        # [bsize, seq_len, hidden_size]
        if self.linear is not None:
            embeddings = self.linear(embeddings)
        # [bsize, seq_len, num_values]
        # value_scores = torch.matmul(embeddings, self.value_embeddings.weight.t())
        # [bsize, seq_len, 1, out_dim] * [1, 1, num_values, out_dim] -> [bsize, seq_len, num_values, out_dim]
        value_scores = embeddings.unsqueeze(dim=2) * self.value_embeddings.weight.unsqueeze(dim=0).unsqueeze(dim=0)
        # [bsize, seq_len, num_values]
        value_scores = value_scores.sum(dim=-1)
        # [bsize, seq_len, num_values] + [bsize, seq_len, 1]
        value_scores = value_scores * (1.0 - embeddings_mask.unsqueeze(dim=-1).float())
        # [bsize, num_values]
        value_scores = value_scores.sum(dim=1)
        # [bsize, num_values]
        value_scores = value_scores + (1.0 - value_mask.float()) * -1e9
        # [bsize, num_values]
        value_probs = self.value_softmax(value_scores)
        # [1, num_values, out_dim] * [bsize, num_values, 1]
        value_embeddings = self.value_embeddings.weight.unsqueeze(dim=0) * value_probs.unsqueeze(dim=-1)
        # [bsize, out_dim]
        value_embeddings = value_embeddings.sum(dim=1)
