from abc import ABC, abstractmethod
from typing import List

import torch.nn as nn


class MultiValuesModule(nn.Module, ABC):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cultural_list: List[str],
        moral_list: List[str],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cultural_list = cultural_list
        self.num_cultural_values = len(cultural_list)
        self.moral_list = moral_list
        self.num_moral_values = len(moral_list)
        self.cultural_embeddings = nn.Embedding(self.num_cultural_values, self.output_dim)
        self.moral_embeddings = nn.Embedding(self.num_moral_values, self.output_dim)

    @abstractmethod
    def forward(self, embeddings, embeddings_mask, cultural_mask, moral_mask, post_mask=None, frame_mask=None):
        pass


class MultiValuesWeightedAttentionPooling(MultiValuesModule):
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
        self.seq_softmax = nn.Softmax(dim=-1)
        self.val_seq_softmax = nn.Softmax(dim=-2)

    def combine_masks(self, mask_a, mask_b):
        # 0 x 0 = 0
        # 0 x 1 = 0
        # 1 x 0 = 0
        # 1 x 1 = 1
        return mask_a * mask_b

    def sequence_attention(self, embeddings, embeddings_mask, value_mask, value_embeddings):
        # [bsize, seq_len, 1, out_dim] x [1, 1, num_values, out_dim] -> [bsize, seq_len, num_values, out_dim]
        value_seq_scores = embeddings.unsqueeze(dim=2) * value_embeddings.weight.unsqueeze(dim=0).unsqueeze(dim=0)
        # [bsize, seq_len, num_values]
        value_seq_scores = value_seq_scores.sum(dim=-1)
        # [bsize, seq_len, num_values] * [bsize, 1, num_values]
        value_seq_scores = value_seq_scores * value_mask.unsqueeze(dim=1).float()
        # [bsize, seq_len]
        seq_scores = value_seq_scores.sum(dim=-1)
        # [bsize, seq_len] + [bsize, seq_len]
        seq_scores = seq_scores + (1.0 - embeddings_mask.float()) * -1e9
        # [bsize, seq_len]
        seq_probs = self.seq_softmax(seq_scores)
        # [bsize, seq_len, 1] * [bsize, seq_len, out_dim] -> [bsize, seq_len, out_dim]
        seq_embeddings = seq_probs.unsqueeze(dim=-1) * embeddings
        # [bsize, out_dim]
        seq_embedding = seq_embeddings.sum(dim=1)
        return seq_embedding, seq_probs

    def sequence_value_attention(self, embeddings, embeddings_mask, value_mask, value_embeddings):
        # [bsize, seq_len, 1, out_dim] x [1, 1, num_values, out_dim] -> [bsize, seq_len, num_values, out_dim]
        seq_value_scores = embeddings.unsqueeze(dim=2) * value_embeddings.weight.unsqueeze(dim=0).unsqueeze(dim=0)
        # [bsize, seq_len, num_values]
        seq_value_scores = seq_value_scores.sum(dim=-1)
        # [bsize, seq_len, num_values] * [bsize, 1, num_values]
        seq_value_scores = seq_value_scores + (1.0 - value_mask.unsqueeze(dim=1).float()) * -1e9
        # [bsize, seq_len, num_values] * [bsize, seq_len, 1]
        seq_value_scores = seq_value_scores + (1.0 - embeddings_mask.unsqueeze(dim=-1).float()) * -1e9
        # [bsize, seq_len, num_values]
        seq_value_probs = self.value_softmax(seq_value_scores)
        # [bsize, seq_len, num_values, out_dim] * [bsize, seq_len, num_values, 1]
        # -> [bsize, seq_len, num_values, out_dim]
        embs = value_embeddings.weight.unsqueeze(dim=0).unsqueeze(dim=0)
        seq_value_embeddings = seq_value_probs.unsqueeze(dim=-1) * embs
        # [bsize, seq_len, out_dim]
        seq_value_embedding = seq_value_embeddings.sum(dim=2)
        return seq_value_embedding, seq_value_probs

    def value_sequence_attention(self, embeddings, embeddings_mask, value_mask, value_embeddings):
        # [bsize, seq_len, 1, out_dim] x [1, 1, num_values, out_dim] -> [bsize, seq_len, num_values, out_dim]
        seq_value_scores = embeddings.unsqueeze(dim=2) * value_embeddings.weight.unsqueeze(dim=0).unsqueeze(dim=0)
        # [bsize, seq_len, num_values]
        seq_value_scores = seq_value_scores.sum(dim=-1)
        # [bsize, seq_len, num_values] * [bsize, 1, num_values]
        seq_value_scores = seq_value_scores + (1.0 - value_mask.unsqueeze(dim=1).float()) * -1e9
        # [bsize, seq_len, num_values] * [bsize, seq_len, 1]
        seq_value_scores = seq_value_scores + (1.0 - embeddings_mask.unsqueeze(dim=-1).float()) * -1e9
        # [bsize, seq_len, num_values]
        value_seq_probs = self.val_seq_softmax(seq_value_scores)
        # [bsize, seq_len, num_values, out_dim] * [bsize, seq_len, num_values, 1]
        # -> [bsize, seq_len, num_values, out_dim]
        value_seq_embeddings = value_seq_probs.unsqueeze(dim=-1) * embeddings.unsqueeze(dim=2)
        # [bsize, num_values, out_dim]
        value_seq_embedding = value_seq_embeddings.sum(dim=1)
        return value_seq_embedding, value_seq_probs

    def value_attention(self, embeddings, embeddings_mask, value_mask, value_embeddings):
        # [bsize, seq_len, num_values]
        # value_scores = torch.matmul(embeddings, self.value_embeddings.weight.t())
        # [bsize, seq_len, 1, out_dim] * [1, 1, num_values, out_dim] -> [bsize, seq_len, num_values, out_dim]
        value_scores = embeddings.unsqueeze(dim=2) * value_embeddings.weight.unsqueeze(dim=0).unsqueeze(dim=0)
        # [bsize, seq_len, num_values]
        value_scores = value_scores.sum(dim=-1)
        # [bsize, seq_len, num_values] + [bsize, seq_len, 1]
        value_scores = value_scores * embeddings_mask.unsqueeze(dim=-1).float()
        # [bsize, num_values]
        value_scores = value_scores.sum(dim=1)
        # [bsize, num_values]
        value_scores = value_scores + (1.0 - value_mask.float()) * -1e9
        # [bsize, num_values]
        value_probs = self.value_softmax(value_scores)
        # [1, num_values, out_dim] * [bsize, num_values, 1]
        value_embeddings = value_embeddings.weight.unsqueeze(dim=0) * value_probs.unsqueeze(dim=-1)
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

    def forward(self, embeddings, embeddings_mask, cultural_mask, moral_mask, post_mask, frame_mask):
        embeddings = self.preprocess(embeddings)

        post_mask = self.combine_masks(embeddings_mask, post_mask)

        frame_mask = self.combine_masks(embeddings_mask, frame_mask)

        # value attention - which value info matters for frame
        frame_cultural, frame_cultural_probs = self.value_attention(
            embeddings, frame_mask, cultural_mask, self.cultural_embeddings
        )
        frame_moral, frame_moral_probs = self.value_attention(embeddings, frame_mask, moral_mask, self.moral_embeddings)

        # attention pooling - which post info matters for values
        post_cultural, post_cultural_probs = self.attention_pooling(embeddings, post_mask, frame_cultural)
        post_moral, post_moral_probs = self.attention_pooling(embeddings, post_mask, frame_moral)

        pooled_embedding = post_cultural + post_moral

        return {
            "frame_cultural": frame_cultural,
            "frame_moral": frame_moral,
            "frame_cultural_probs": frame_cultural_probs,
            "frame_moral_probs": frame_moral_probs,
            "post_cultural": post_cultural,
            "post_moral": post_moral,
            "post_cultural_probs": post_cultural_probs,
            "post_moral_probs": post_moral_probs,
            "pooled_embedding": pooled_embedding,
            "output_features": pooled_embedding,
        }


class MultiValuesExtraWeightedAttentionPooling(MultiValuesWeightedAttentionPooling):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.moral_linear = nn.Linear(self.input_dim, self.output_dim)

    def preprocess(self, embeddings, layer):
        # [bsize, seq_len, hidden_size] -> [bsize, seq_len, out_dim]
        if self.linear is not None:
            # TODO add dropout
            embeddings = self.linear(embeddings)
        return embeddings

    def forward(self, embeddings, embeddings_mask, cultural_mask, moral_mask, post_mask, frame_mask):
        cultural_embeddings = self.linear(embeddings)
        moral_embeddings = self.moral_linear(embeddings)

        post_mask = self.combine_masks(embeddings_mask, post_mask)

        frame_mask = self.combine_masks(embeddings_mask, frame_mask)

        # value attention - which value info matters for frame
        frame_cultural, frame_cultural_probs = self.value_attention(
            cultural_embeddings, frame_mask, cultural_mask, self.cultural_embeddings
        )
        frame_moral, frame_moral_probs = self.value_attention(
            moral_embeddings, frame_mask, moral_mask, self.moral_embeddings
        )

        # attention pooling - which post info matters for values
        post_cultural, post_cultural_probs = self.attention_pooling(cultural_embeddings, post_mask, frame_cultural)
        post_moral, post_moral_probs = self.attention_pooling(moral_embeddings, post_mask, frame_moral)

        pooled_embedding = post_cultural + post_moral

        return {
            "frame_cultural": frame_cultural,
            "frame_moral": frame_moral,
            "frame_cultural_probs": frame_cultural_probs,
            "frame_moral_probs": frame_moral_probs,
            "post_cultural": post_cultural,
            "post_moral": post_moral,
            "post_cultural_probs": post_cultural_probs,
            "post_moral_probs": post_moral_probs,
            "pooled_embedding": pooled_embedding,
            "output_features": pooled_embedding,
        }


class MultiValuesQueryValueAttentionPooling(MultiValuesWeightedAttentionPooling):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.moral_query = nn.Linear(self.input_dim, self.output_dim)
        self.moral_value = nn.Linear(self.input_dim, self.output_dim)
        self.cultural_value = nn.Linear(self.input_dim, self.output_dim)

    def value_attention(self, embeddings, embeddings_mask, value_mask, value_embeddings):
        # [bsize, seq_len, num_values]
        # value_scores = torch.matmul(embeddings, self.value_embeddings.weight.t())
        # [bsize, seq_len, 1, out_dim] * [1, 1, num_values, out_dim] -> [bsize, seq_len, num_values, out_dim]
        value_scores = embeddings.unsqueeze(dim=2) * value_embeddings.weight.unsqueeze(dim=0).unsqueeze(dim=0)
        # [bsize, seq_len, num_values]
        value_scores = value_scores.sum(dim=-1)
        # [bsize, seq_len, num_values] + [bsize, seq_len, 1]
        value_scores = value_scores * embeddings_mask.unsqueeze(dim=-1).float()
        # [bsize, num_values]
        value_scores = value_scores.sum(dim=1)
        # [bsize, num_values]
        value_scores = value_scores + (1.0 - value_mask.float()) * -1e9
        # [bsize, num_values]
        value_probs = self.value_softmax(value_scores / (embeddings.shape[-1]) ** 0.5)
        # [1, num_values, out_dim] * [bsize, num_values, 1]
        value_embeddings = value_embeddings.weight.unsqueeze(dim=0) * value_probs.unsqueeze(dim=-1)
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
        pooling_probs = self.pool_softmax(pooling_scores / (embeddings.shape[-1]) ** 0.5)
        # [bsize, seq_len, out_dim] * [bsize, seq_len, 1] -> [bsize, seq_len, out_dim]
        pooled_embeddings = embeddings * pooling_probs.unsqueeze(dim=-1)
        # [bsize, out_dim]
        pooled_embedding = pooled_embeddings.sum(dim=1)
        return pooled_embedding, pooling_probs

    def forward(self, embeddings, embeddings_mask, cultural_mask, moral_mask, post_mask, frame_mask):
        cultural_queries = self.linear(embeddings)
        moral_queries = self.moral_query(embeddings)
        cultural_values = self.cultural_value(embeddings)
        moral_values = self.moral_value(embeddings)

        post_mask = self.combine_masks(embeddings_mask, post_mask)

        frame_mask = self.combine_masks(embeddings_mask, frame_mask)

        # value attention - which value info matters for frame
        frame_cultural, frame_cultural_probs = self.value_attention(
            cultural_queries, frame_mask, cultural_mask, self.cultural_embeddings
        )
        frame_moral, frame_moral_probs = self.value_attention(
            moral_queries, frame_mask, moral_mask, self.moral_embeddings
        )

        # attention pooling - which post info matters for values
        post_cultural, post_cultural_probs = self.attention_pooling(cultural_values, post_mask, frame_cultural)
        post_moral, post_moral_probs = self.attention_pooling(moral_values, post_mask, frame_moral)

        pooled_embedding = post_cultural + post_moral

        return {
            "frame_cultural": frame_cultural,
            "frame_moral": frame_moral,
            "frame_cultural_probs": frame_cultural_probs,
            "frame_moral_probs": frame_moral_probs,
            "post_cultural": post_cultural,
            "post_moral": post_moral,
            "post_cultural_probs": post_cultural_probs,
            "post_moral_probs": post_moral_probs,
            "pooled_embedding": pooled_embedding,
            "output_features": pooled_embedding,
        }


class MultiValuesQueryValueDiffAttentionPooling(MultiValuesQueryValueAttentionPooling):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, embeddings_mask, cultural_mask, moral_mask, post_mask, frame_mask):
        cultural_queries = self.linear(embeddings)
        moral_queries = self.moral_query(embeddings)
        cultural_values = self.cultural_value(embeddings)
        moral_values = self.moral_value(embeddings)

        post_mask = self.combine_masks(embeddings_mask, post_mask)

        frame_mask = self.combine_masks(embeddings_mask, frame_mask)

        # value attention - which value info matters for frame
        frame_cultural, frame_cultural_probs = self.value_attention(
            cultural_queries, frame_mask, cultural_mask, self.cultural_embeddings
        )
        frame_moral, frame_moral_probs = self.value_attention(
            moral_queries, frame_mask, moral_mask, self.moral_embeddings
        )

        # attention pooling - which post info matters for values
        post_cultural, post_cultural_probs = self.attention_pooling(cultural_values, post_mask, frame_cultural)
        post_moral, post_moral_probs = self.attention_pooling(moral_values, post_mask, frame_moral)

        pooled_embedding = post_cultural - post_moral

        return {
            "frame_cultural": frame_cultural,
            "frame_moral": frame_moral,
            "frame_cultural_probs": frame_cultural_probs,
            "frame_moral_probs": frame_moral_probs,
            "post_cultural": post_cultural,
            "post_moral": post_moral,
            "post_cultural_probs": post_cultural_probs,
            "post_moral_probs": post_moral_probs,
            "pooled_embedding": pooled_embedding,
            "output_features": pooled_embedding,
        }


class MultiValuesQueryValueCentroidAttentionPooling(MultiValuesQueryValueAttentionPooling):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, embeddings_mask, cultural_mask, moral_mask, post_mask, frame_mask):
        cultural_queries = self.linear(embeddings)
        moral_queries = self.moral_query(embeddings)
        cultural_values = self.cultural_value(embeddings)
        moral_values = self.moral_value(embeddings)

        post_mask = self.combine_masks(embeddings_mask, post_mask)

        frame_mask = self.combine_masks(embeddings_mask, frame_mask)

        # value attention - which value info matters for frame
        frame_cultural, frame_cultural_probs = self.value_attention(
            cultural_queries, frame_mask, cultural_mask, self.cultural_embeddings
        )
        frame_moral, frame_moral_probs = self.value_attention(
            moral_queries, frame_mask, moral_mask, self.moral_embeddings
        )

        # attention pooling - which post info matters for values
        post_cultural, post_cultural_probs = self.attention_pooling(cultural_values, post_mask, frame_cultural)
        post_moral, post_moral_probs = self.attention_pooling(moral_values, post_mask, frame_moral)

        pooled_embedding = (post_cultural + post_moral) / 2.0

        return {
            "frame_cultural": frame_cultural,
            "frame_moral": frame_moral,
            "frame_cultural_probs": frame_cultural_probs,
            "frame_moral_probs": frame_moral_probs,
            "post_cultural": post_cultural,
            "post_moral": post_moral,
            "post_cultural_probs": post_cultural_probs,
            "post_moral_probs": post_moral_probs,
            "pooled_embedding": pooled_embedding,
            "output_features": pooled_embedding,
        }


class MultiValuesQueryValueDotAttentionPooling(MultiValuesQueryValueAttentionPooling):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, embeddings_mask, cultural_mask, moral_mask, post_mask, frame_mask):
        cultural_queries = self.linear(embeddings)
        moral_queries = self.moral_query(embeddings)
        cultural_values = self.cultural_value(embeddings)
        moral_values = self.moral_value(embeddings)

        post_mask = self.combine_masks(embeddings_mask, post_mask)

        frame_mask = self.combine_masks(embeddings_mask, frame_mask)

        # value attention - which value info matters for frame
        frame_cultural, frame_cultural_probs = self.value_attention(
            cultural_queries, frame_mask, cultural_mask, self.cultural_embeddings
        )
        frame_moral, frame_moral_probs = self.value_attention(
            moral_queries, frame_mask, moral_mask, self.moral_embeddings
        )

        # attention pooling - which post info matters for values
        post_cultural, post_cultural_probs = self.attention_pooling(cultural_values, post_mask, frame_cultural)
        post_moral, post_moral_probs = self.attention_pooling(moral_values, post_mask, frame_moral)

        pooled_embedding = post_cultural * post_moral

        return {
            "frame_cultural": frame_cultural,
            "frame_moral": frame_moral,
            "frame_cultural_probs": frame_cultural_probs,
            "frame_moral_probs": frame_moral_probs,
            "post_cultural": post_cultural,
            "post_moral": post_moral,
            "post_cultural_probs": post_cultural_probs,
            "post_moral_probs": post_moral_probs,
            "pooled_embedding": pooled_embedding,
            "output_features": pooled_embedding,
        }


class MultiValuesQueryValueDistAttentionPooling(MultiValuesQueryValueAttentionPooling):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, embeddings_mask, cultural_mask, moral_mask, post_mask, frame_mask):
        cultural_queries = self.linear(embeddings)
        moral_queries = self.moral_query(embeddings)
        cultural_values = self.cultural_value(embeddings)
        moral_values = self.moral_value(embeddings)

        post_mask = self.combine_masks(embeddings_mask, post_mask)

        frame_mask = self.combine_masks(embeddings_mask, frame_mask)

        # value attention - which value info matters for frame
        frame_cultural, frame_cultural_probs = self.value_attention(
            cultural_queries, frame_mask, cultural_mask, self.cultural_embeddings
        )
        frame_moral, frame_moral_probs = self.value_attention(
            moral_queries, frame_mask, moral_mask, self.moral_embeddings
        )

        # attention pooling - which post info matters for values
        post_cultural, post_cultural_probs = self.attention_pooling(cultural_values, post_mask, frame_cultural)
        post_moral, post_moral_probs = self.attention_pooling(moral_values, post_mask, frame_moral)

        pooled_embedding = (post_cultural - post_moral) ** 2

        return {
            "frame_cultural": frame_cultural,
            "frame_moral": frame_moral,
            "frame_cultural_probs": frame_cultural_probs,
            "frame_moral_probs": frame_moral_probs,
            "post_cultural": post_cultural,
            "post_moral": post_moral,
            "post_cultural_probs": post_cultural_probs,
            "post_moral_probs": post_moral_probs,
            "pooled_embedding": pooled_embedding,
            "output_features": pooled_embedding,
        }


class MultiValuesQueryValueMultiHeadDiffAttentionPooling(MultiValuesModule):
    def __init__(
        self,
        num_heads: int,
        dropout: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.moral_attention = nn.MultiheadAttention(
            embed_dim=self.output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cultural_attention = nn.MultiheadAttention(
            embed_dim=self.output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        if self.input_dim != self.output_dim:
            self.linear = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.linear = None

    def combine_masks(self, mask_a, mask_b):
        # 0 x 0 = 0
        # 0 x 1 = 0
        # 1 x 0 = 0
        # 1 x 1 = 1
        return mask_a * mask_b

    def preprocess(self, embeddings):
        # [bsize, seq_len, hidden_size] -> [bsize, seq_len, out_dim]
        if self.linear is not None:
            # TODO add dropout
            embeddings = self.linear(embeddings)
        return embeddings

    def forward(self, embeddings, embeddings_mask, cultural_mask, moral_mask, post_mask, frame_mask):
        embeddings = self.preprocess(embeddings)

        post_mask = self.combine_masks(embeddings_mask, post_mask)
        frame_mask = self.combine_masks(embeddings_mask, frame_mask)

        # cultural
        # [bsize, seq_len, out_dim]
        # [bize, seq_len, num_values]
        frame_cultural_embeddings, frame_cultural_weights = self.cultural_attention(
            # [bsize, seq_len, out_dim]
            query=embeddings,
            # [bsize, num_values, out_dim]
            key=self.cultural_embeddings.weight.unsqueeze(dim=0).repeat(embeddings.shape[0], 1, 1),
            # [bsize, num_values, out_dim]
            value=self.cultural_embeddings.weight.unsqueeze(dim=0).repeat(embeddings.shape[0], 1, 1),
            need_weights=True,
            average_attn_weights=True,
            # [bsize, num_values]
            key_padding_mask=cultural_mask.float(),
            # [bsize, seq_len] -> [bsize, num_heads, seq_len, 1] -> [bsize * num_heads, seq_len, 1]
            attn_mask=frame_mask.unsqueeze(dim=1)
            .unsqueeze(dim=-1)
            .repeat(1, self.num_heads, 1, self.num_cultural_values)
            .view(-1, embeddings.shape[1], self.num_cultural_values)
            .float(),
        )

        # option 1: contextualize the post now
        # [bsize, seq_len, out_dim]
        # [bize, seq_len, num_values]
        post_cultural_embeddings, post_cultural_weights = self.cultural_attention(
            # [bsize, seq_len, out_dim]
            query=frame_cultural_embeddings,
            # [bsize, num_values, out_dim]
            key=self.cultural_embeddings.weight.unsqueeze(dim=0).repeat(embeddings.shape[0], 1, 1),
            # [bsize, num_values, out_dim]
            value=self.cultural_embeddings.weight.unsqueeze(dim=0).repeat(embeddings.shape[0], 1, 1),
            need_weights=True,
            average_attn_weights=True,
            # [bsize, num_values]
            key_padding_mask=cultural_mask.float(),
            # [bsize, seq_len] -> [bsize, num_heads, seq_len, 1] -> [bsize * num_heads, seq_len, 1]
            attn_mask=post_mask.unsqueeze(dim=1)
            .unsqueeze(dim=-1)
            .repeat(1, self.num_heads, 1, self.num_cultural_values)
            .view(-1, embeddings.shape[1], self.num_cultural_values)
            .float(),
        )

        # [bsize, out_dim]
        post_cultural = post_cultural_embeddings.sum(dim=1) / post_mask.sum(dim=1, keepdim=True).float()

        # moral
        # [bsize, seq_len, out_dim]
        # [bize, seq_len, num_values]
        frame_moral_embeddings, frame_moral_weights = self.moral_attention(
            # [bsize, seq_len, out_dim]
            query=embeddings,
            # [bsize, num_values, out_dim]
            key=self.moral_embeddings.weight.unsqueeze(dim=0).repeat(embeddings.shape[0], 1, 1),
            # [bsize, num_values, out_dim]
            value=self.moral_embeddings.weight.unsqueeze(dim=0).repeat(embeddings.shape[0], 1, 1),
            need_weights=True,
            average_attn_weights=True,
            # [bsize, num_values]
            key_padding_mask=moral_mask.float(),
            # [bsize, seq_len] -> [bsize, num_heads, seq_len, 1] -> [bsize * num_heads, seq_len, 1]
            attn_mask=frame_mask.unsqueeze(dim=1)
            .unsqueeze(dim=-1)
            .repeat(1, self.num_heads, 1, self.num_moral_values)
            .view(-1, embeddings.shape[1], self.num_moral_values)
            .float(),
        )

        # option 1: contextualize the post now
        # [bsize, seq_len, out_dim]
        # [bize, seq_len, num_values]
        post_moral_embeddings, post_moral_weights = self.moral_attention(
            # [bsize, seq_len, out_dim]
            query=frame_moral_embeddings,
            # [bsize, num_values, out_dim]
            key=self.moral_embeddings.weight.unsqueeze(dim=0).repeat(embeddings.shape[0], 1, 1),
            # [bsize, num_values, out_dim]
            value=self.moral_embeddings.weight.unsqueeze(dim=0).repeat(embeddings.shape[0], 1, 1),
            need_weights=True,
            average_attn_weights=True,
            # [bsize, num_values]
            key_padding_mask=moral_mask.float(),
            # [bsize, seq_len] -> [bsize, num_heads, seq_len, 1] -> [bsize * num_heads, seq_len, 1]
            attn_mask=post_mask.unsqueeze(dim=1)
            .unsqueeze(dim=-1)
            .repeat(1, self.num_heads, 1, self.num_moral_values)
            .view(-1, embeddings.shape[1], self.num_moral_values)
            .float(),
        )

        # [bsize, out_dim]
        post_moral = post_moral_embeddings.sum(dim=1) / post_mask.sum(dim=1, keepdim=True).float()

        pooled_embedding = post_cultural - post_moral

        return {
            "post_cultural": post_cultural,
            "pooled_embedding": pooled_embedding,
            "output_features": pooled_embedding,
        }


class MultiValuesQueryValueMultiHeadDiffCrossAttentionPooling(MultiValuesModule):
    def __init__(
        self,
        num_heads: int,
        dropout: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.moral_attention = nn.MultiheadAttention(
            embed_dim=self.output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cultural_attention = nn.MultiheadAttention(
            embed_dim=self.output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        if self.input_dim != self.output_dim:
            self.linear = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.linear = None

    def combine_masks(self, mask_a, mask_b):
        # 0 x 0 = 0
        # 0 x 1 = 0
        # 1 x 0 = 0
        # 1 x 1 = 1
        return mask_a * mask_b

    def preprocess(self, embeddings):
        # [bsize, seq_len, hidden_size] -> [bsize, seq_len, out_dim]
        if self.linear is not None:
            # TODO add dropout
            embeddings = self.linear(embeddings)
        return embeddings

    def forward(self, embeddings, embeddings_mask, cultural_mask, moral_mask, post_mask, frame_mask):
        embeddings = self.preprocess(embeddings)

        post_mask = self.combine_masks(embeddings_mask, post_mask)
        frame_mask = self.combine_masks(embeddings_mask, frame_mask)

        # cultural
        # [bsize, seq_len, out_dim]
        # [bize, seq_len, num_values]
        frame_cultural_embeddings, frame_cultural_weights = self.cultural_attention(
            # [bsize, seq_len, out_dim]
            query=embeddings,
            # [bsize, num_values, out_dim]
            key=self.cultural_embeddings.weight.unsqueeze(dim=0).repeat(embeddings.shape[0], 1, 1),
            # [bsize, num_values, out_dim]
            value=self.cultural_embeddings.weight.unsqueeze(dim=0).repeat(embeddings.shape[0], 1, 1),
            need_weights=True,
            average_attn_weights=True,
            # [bsize, num_values]
            key_padding_mask=cultural_mask.float(),
            # [bsize, seq_len] -> [bsize, num_heads, seq_len, 1] -> [bsize * num_heads, seq_len, 1]
            attn_mask=frame_mask.unsqueeze(dim=1)
            .unsqueeze(dim=-1)
            .repeat(1, self.num_heads, 1, self.num_cultural_values)
            .view(-1, embeddings.shape[1], self.num_cultural_values)
            .float(),
        )

        # option 2: contextualize the post with the frame
        # [bsize, seq_len, out_dim]
        # [bize, seq_len, seq_len]
        post_cultural_embeddings, post_cultural_weights = self.cultural_attention(
            # [bsize, seq_len, out_dim]
            query=frame_cultural_embeddings,
            # [bsize, seq_len, out_dim]
            key=embeddings,
            # [bsize, seq_len, out_dim]
            value=embeddings,
            need_weights=True,
            average_attn_weights=True,
            # [bsize, seq_len]
            key_padding_mask=post_mask.float(),
            # [bsize, seq_len] -> [bsize, num_heads, seq_len, 1] -> [bsize * num_heads, seq_len, 1]
            attn_mask=frame_mask.unsqueeze(dim=1)
            .unsqueeze(dim=-1)
            .repeat(1, self.num_heads, 1, self.num_cultural_values)
            .view(-1, embeddings.shape[1], self.num_cultural_values)
            .float(),
        )

        # [bsize, out_dim]
        post_cultural = post_cultural_embeddings.sum(dim=1) / post_mask.sum(dim=1, keepdim=True).float()

        # moral
        # [bsize, seq_len, out_dim]
        # [bize, seq_len, num_values]
        frame_moral_embeddings, frame_moral_weights = self.moral_attention(
            # [bsize, seq_len, out_dim]
            query=embeddings,
            # [bsize, num_values, out_dim]
            key=self.moral_embeddings.weight.unsqueeze(dim=0).repeat(embeddings.shape[0], 1, 1),
            # [bsize, num_values, out_dim]
            value=self.moral_embeddings.weight.unsqueeze(dim=0).repeat(embeddings.shape[0], 1, 1),
            need_weights=True,
            average_attn_weights=True,
            # [bsize, num_values]
            key_padding_mask=moral_mask.float(),
            # [bsize, seq_len] -> [bsize, num_heads, seq_len, 1] -> [bsize * num_heads, seq_len, 1]
            attn_mask=frame_mask.unsqueeze(dim=1)
            .unsqueeze(dim=-1)
            .repeat(1, self.num_heads, 1, self.num_moral_values)
            .view(-1, embeddings.shape[1], self.num_moral_values)
            .float(),
        )

        # option 2: contextualize the post with the frame
        # [bsize, seq_len, out_dim]
        # [bize, seq_len, seq_len]
        post_moral_embeddings, post_moral_weights = self.moral_attention(
            # [bsize, seq_len, out_dim]
            query=frame_moral_embeddings,
            # [bsize, seq_len, out_dim]
            key=embeddings,
            # [bsize, seq_len, out_dim]
            value=embeddings,
            need_weights=True,
            average_attn_weights=True,
            # [bsize, seq_len]
            key_padding_mask=post_mask.float(),
            # [bsize, seq_len] -> [bsize, num_heads, seq_len, 1] -> [bsize * num_heads, seq_len, 1]
            attn_mask=frame_mask.unsqueeze(dim=1)
            .unsqueeze(dim=-1)
            .repeat(1, self.num_heads, 1, self.num_moral_values)
            .view(-1, embeddings.shape[1], self.num_moral_values)
            .float(),
        )

        # [bsize, out_dim]
        post_moral = post_moral_embeddings.sum(dim=1) / post_mask.sum(dim=1, keepdim=True).float()

        pooled_embedding = post_cultural - post_moral

        return {
            "post_cultural": post_cultural,
            "pooled_embedding": pooled_embedding,
            "output_features": pooled_embedding,
        }
