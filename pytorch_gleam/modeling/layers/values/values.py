from abc import ABC, abstractmethod
from typing import List

import torch.nn as nn


class ValuesModule(nn.module, ABC):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        value_list: List[str],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.value_list = value_list
        self.num_values = len(value_list)
        self.value_embeddings = nn.Embedding(self.num_values, self.output_dim)

    @abstractmethod
    def forward(self, embeddings, embeddings_mask, value_mask, post_mask=None, frame_mask=None):
        pass


class ValuesAttentionPooling(ValuesModule):
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
        self.pool_softmax = nn.Softmax(dim=-1)

    def forward(self, embeddings, embeddings_mask, value_mask, post_mask=None, frame_mask=None):
        # [bsize, seq_len, hidden_size] -> [bsize, seq_len, out_dim]
        if self.linear is not None:
            # TODO add dropout
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
        value_embedding = value_embeddings.sum(dim=1)

        # attention pooling
        # [bsize, seq_len, out_dim] * [bsize, 1, out_dim] -> [bsize, seq_len, out_dim]
        pooling_scores = embeddings * value_embedding.unsqueeze(dim=1)
        # [bsize, seq_len, out_dim] -> [bsize, seq_len]
        pooling_scores = pooling_scores.sum(dim=-1)
        # [bsize, seq_len] + [bsize, seq_len]
        pooling_scores = pooling_scores + (1.0 - embeddings_mask.float()) * -1e9
        # [bsize, seq_len]
        pooling_probs = self.pool_softmax(pooling_scores)
        # [bsize, seq_len, out_dim] * [bsize, seq_len, 1] -> [bsize, seq_len, out_dim]
        pooled_embeddings = embeddings * pooling_probs.unsqueeze(dim=-1)
        # [bsize, out_dim]
        pooled_embedding = pooled_embeddings.sum(dim=1)
        return {
            "value_embedding": value_embedding,
            "pooled_embedding": pooled_embedding,
            "value_probs": value_probs,
            "pooling_probs": pooling_probs,
            "output_features": pooled_embedding,
        }
