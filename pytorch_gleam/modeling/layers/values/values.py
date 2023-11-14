from abc import ABC, abstractmethod
from typing import List

import torch.nn as nn


class ValuesModule(nn.Module, ABC):
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

    def combine_masks(self, mask_a, mask_b):
        # 0 x 0 = 0
        # 0 x 1 = 0
        # 1 x 0 = 0
        # 1 x 1 = 1
        return mask_a * mask_b

    def value_attention(self, embeddings, embeddings_mask, value_mask):
        # [bsize, seq_len, num_values]
        # value_scores = torch.matmul(embeddings, self.value_embeddings.weight.t())
        # [bsize, seq_len, 1, out_dim] * [1, 1, num_values, out_dim] -> [bsize, seq_len, num_values, out_dim]
        value_scores = embeddings.unsqueeze(dim=2) * self.value_embeddings.weight.unsqueeze(dim=0).unsqueeze(dim=0)
        # [bsize, seq_len, num_values]
        value_scores = value_scores.sum(dim=-1)
        # [bsize, seq_len, num_values] * [bsize, seq_len, 1]
        value_scores = value_scores * embeddings_mask.unsqueeze(dim=-1).float()
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
        return value_embedding, value_probs

    def attention_pooling(self, embeddings, embeddings_mask, value_embedding):
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
        return pooled_embedding, pooling_probs

    def preprocess(self, embeddings):
        # [bsize, seq_len, hidden_size] -> [bsize, seq_len, out_dim]
        if self.linear is not None:
            # TODO add dropout
            embeddings = self.linear(embeddings)
        return embeddings

    def forward(self, embeddings, embeddings_mask, value_mask, post_mask=None, frame_mask=None):
        embeddings = self.preprocess(embeddings)
        # value attention
        value_embedding, value_probs = self.value_attention(embeddings, embeddings_mask, value_mask)

        # attention pooling
        pooled_embedding, pooling_probs = self.attention_pooling(embeddings, embeddings_mask, value_embedding)

        return {
            "value_embedding": value_embedding,
            "pooled_embedding": pooled_embedding,
            "value_probs": value_probs,
            "pooling_probs": pooling_probs,
            "output_features": pooled_embedding,
        }


class ValuesDifferenceAttentionPooling(ValuesAttentionPooling):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, embeddings_mask, value_mask, post_mask, frame_mask):
        embeddings = self.preprocess(embeddings)

        post_mask = self.combine_masks(embeddings_mask, post_mask)
        # post value attention
        post_value_embedding, post_value_probs = self.value_attention(embeddings, post_mask, value_mask)

        frame_mask = self.combine_masks(embeddings_mask, frame_mask)
        # frame value attention
        frame_value_embedding, frame_value_probs = self.value_attention(embeddings, frame_mask, value_mask)

        # diff
        value_embedding = frame_value_embedding - post_value_embedding

        # attention pooling
        pooled_embedding, pooling_probs = self.attention_pooling(embeddings, embeddings_mask, value_embedding)

        return {
            "value_embedding": value_embedding,
            "pooled_embedding": pooled_embedding,
            "post_value_probs": post_value_probs,
            "frame_value_probs": frame_value_probs,
            "pooling_probs": pooling_probs,
            "output_features": pooled_embedding,
        }


class ValuesAddedAttentionPooling(ValuesAttentionPooling):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, embeddings_mask, value_mask, post_mask, frame_mask):
        embeddings = self.preprocess(embeddings)

        post_mask = self.combine_masks(embeddings_mask, post_mask)
        # post value attention
        post_value_embedding, post_value_probs = self.value_attention(embeddings, post_mask, value_mask)

        frame_mask = self.combine_masks(embeddings_mask, frame_mask)
        # frame value attention
        frame_value_embedding, frame_value_probs = self.value_attention(embeddings, frame_mask, value_mask)

        # diff
        value_embedding = frame_value_embedding + post_value_embedding

        # attention pooling
        pooled_embedding, pooling_probs = self.attention_pooling(embeddings, embeddings_mask, value_embedding)

        return {
            "value_embedding": value_embedding,
            "pooled_embedding": pooled_embedding,
            "post_value_probs": post_value_probs,
            "frame_value_probs": frame_value_probs,
            "pooling_probs": pooling_probs,
            "output_features": pooled_embedding,
        }


class ValuesCentroidAttentionPooling(ValuesAttentionPooling):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, embeddings_mask, value_mask, post_mask, frame_mask):
        embeddings = self.preprocess(embeddings)

        post_mask = self.combine_masks(embeddings_mask, post_mask)
        # post value attention
        post_value_embedding, post_value_probs = self.value_attention(embeddings, post_mask, value_mask)

        frame_mask = self.combine_masks(embeddings_mask, frame_mask)
        # frame value attention
        frame_value_embedding, frame_value_probs = self.value_attention(embeddings, frame_mask, value_mask)

        # diff
        value_embedding = (frame_value_embedding + post_value_embedding) / 2.0

        # attention pooling
        pooled_embedding, pooling_probs = self.attention_pooling(embeddings, embeddings_mask, value_embedding)

        return {
            "value_embedding": value_embedding,
            "pooled_embedding": pooled_embedding,
            "post_value_probs": post_value_probs,
            "frame_value_probs": frame_value_probs,
            "pooling_probs": pooling_probs,
            "output_features": pooled_embedding,
        }


class ValuesWeightedAttentionPooling(ValuesAttentionPooling):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, embeddings_mask, value_mask, post_mask, frame_mask):
        embeddings = self.preprocess(embeddings)

        # value attention
        value_embedding, value_probs = self.value_attention(embeddings, embeddings_mask, value_mask)

        post_mask = self.combine_masks(embeddings_mask, post_mask)

        frame_mask = self.combine_masks(embeddings_mask, frame_mask)

        # attention pooling
        post_embedding, post_probs = self.attention_pooling(embeddings, post_mask, value_embedding)

        frame_embedding, frame_probs = self.attention_pooling(embeddings, frame_mask, value_embedding)

        pooled_embedding = frame_embedding - post_embedding

        return {
            "value_embedding": value_embedding,
            "pooled_embedding": pooled_embedding,
            "value_probs": value_probs,
            "post_probs": post_probs,
            "frame_probs": frame_probs,
            "output_features": pooled_embedding,
        }
