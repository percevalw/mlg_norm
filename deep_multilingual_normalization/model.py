import torch
import torch.nn.functional as F

from nlstruct.modules.search import FAISSIndex, faiss
from nlstruct.utils import torch_global as tg, factorize


def compute_mask_and_shuffle(tokens, mask, min_size, dep_std, masking_rate, keep_first_and_last=True):
    """
    Mask and shuffle function to regularize torch tokens input
    """
    tensor_pos = torch.arange(tokens.shape[1], device=tg.device).float().repeat(tokens.shape[0], 1)
    random_dep = torch.randn(tensor_pos.shape, device=tg.device) * dep_std
    tensor_pos += random_dep
    pad_val = tensor_pos.max() + 2

    tensor_pos.masked_fill_(~mask, pad_val)
    random_mask = torch.rand_like(tensor_pos, device=tg.device, dtype=torch.float)
    if keep_first_and_last:
        random_mask[:, 0] = 1
        random_mask[range(len(tensor_pos)), mask.sum(-1) - 1] = 1
    random_mask.masked_fill_(~mask, 1)
    random_mask[torch.arange(len(random_mask)).unsqueeze(1).repeat(1, min_size), random_mask.topk(min_size, dim=-1, largest=False).indices] = 1
    tensor_pos.masked_fill_(random_mask < masking_rate, pad_val)

    if keep_first_and_last:
        tensor_pos[:, 0] = -float('inf')
        tensor_pos[range(len(tensor_pos)), mask.sum(-1) - 1] = pad_val - 1

    tensor_pos, tensor_pos_sorter = tensor_pos.sort(-1)
    tokens = tokens[torch.arange(tokens.shape[0], device=tg.device).unsqueeze(1), tensor_pos_sorter]
    mask = mask & (tensor_pos != pad_val)
    return tokens, mask


class Metric(torch.nn.Module):
    def __init__(self, in_features, out_features, cluster_label_mask=None, rescale=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("cluster_label_mask", cluster_label_mask)
        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features))
        if rescale is None:
            rescale = torch.nn.Parameter(torch.tensor(20.))
        self.rescale = rescale

        self.reset_parameters()

    @classmethod
    def from_weights(cls, weights):
        instance = cls(weights.shape[0], weights.shape[1])
        instance.weight = torch.nn.Parameter(weights, requires_grad=instance.weight.requires_grad)
        return instance

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, clusters=None, normalize_weight=True, **kwargs):
        x = F.linear(inputs, F.normalize(self.weight, dim=-1) if normalize_weight else self.weight)
        x *= self.rescale
        if clusters is not None and self.cluster_label_mask is not None:
            x = x.masked_fill(~self.cluster_label_mask[clusters], -10000)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, rescale={}, clusters={}'.format(
            self.in_features, self.out_features, self.rescale, self.cluster_label_mask.shape[0] if self.cluster_label_mask is not None else None)


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, cluster_label_mask=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("cluster_label_mask", cluster_label_mask)
        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    @classmethod
    def from_weights(cls, weights):
        instance = cls(weights.shape[0], weights.shape[1])
        instance.weight = torch.nn.Parameter(weights, requires_grad=instance.weight.requires_grad)
        return instance

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, inputs, clusters=None, **kwargs):
        x = F.linear(inputs, self.weight) + self.bias
        if clusters is not None and self.cluster_label_mask is not None:
            x = x.masked_fill(~self.cluster_label_mask[clusters], -10000)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, clusters={}'.format(
            self.in_features, self.out_features, self.cluster_label_mask.shape[0] if self.cluster_label_mask is not None else None)


class FastClusteredIPSearch(object):
    def __init__(self, dim=100, n_clusters=1, use_bias=False, factory_str="Flat", metric=faiss.METRIC_INNER_PRODUCT, nprobe=32, res=None, device=None):
        self.indexes = [FAISSIndex(dim=dim, use_bias=use_bias, factory_str=factory_str, metric=metric, nprobe=nprobe, device=device) for _ in range(n_clusters)]

    def to(self, device, res=None):
        for index in self.indexes:
            index.to(device)

    def train(self, weights, cluster_label_mask, bias=None, and_add=True):
        for cluster, index in enumerate(self.indexes):
            cluster_filter = cluster_label_mask[cluster]
            weights_subset = weights[cluster_filter]
            index = self.indexes[cluster]
            index.train(weights_subset, and_add=and_add, positions=cluster_filter.nonzero(as_tuple=True)[0])
            self.indexes[cluster] = index

    def reset(self):
        for index in self.indexes:
            index.reset()

    def add(self, other, cluster_label_mask):
        for cluster, index in enumerate(self.indexes):
            cluster_filter = cluster_label_mask[cluster]
            other_subset = other[cluster_filter]
            index.add(other_subset, positions=cluster_filter.nonzero(as_tuple=True)[0])

    def search(self, query, clusters, k=1):
        n = query.shape[0]
        rel, _, unique_clusters = factorize(clusters)
        indices = torch.zeros(n, k, dtype=torch.long, device=self.indexes[0].device)
        scores = torch.zeros(n, k, dtype=torch.float, device=self.indexes[0].device)
        for rel_i, cluster in enumerate(unique_clusters.tolist()):
            query_filter = rel == rel_i
            query_subset = query[query_filter]
            scores_subset, indices_subset = self.indexes[cluster].search(query_subset, k=k)
            indices[query_filter] = indices_subset
            scores[query_filter] = scores_subset
        return scores, indices


class Classifier(torch.nn.Module):
    def __init__(self,
                 n_labels,
                 hidden_dim,
                 dropout,
                 n_tokens=None,
                 token_dim=None,
                 embeddings=None,
                 rescale=None,
                 loss='cross_entropy',
                 metric='linear',
                 mask_and_shuffle=(2, 0.5, 0.1),
                 batch_norm_momentum=0.1,
                 batch_norm_affine=True,
                 metric_fc_kwargs=None,
                 ):
        super().__init__()
        if embeddings is not None:
            self.embeddings = embeddings
            if n_tokens is None or token_dim is None:
                if hasattr(embeddings, 'weight'):
                    n_tokens, token_dim = embeddings.weight.shape
                else:
                    n_tokens, token_dim = embeddings.embeddings.weight.shape
        else:
            self.embeddings = torch.nn.Embedding(n_tokens, token_dim) if n_tokens > 0 else None
        assert token_dim is not None, "Provide token_dim or embeddings"
        assert self.embeddings is not None

        dim = (token_dim if n_tokens > 0 else 0)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(dim, hidden_dim)
        metric_fc_kwargs = metric_fc_kwargs if metric_fc_kwargs is not None else {}
        if metric == "linear":
            self.metric_fc = Linear(hidden_dim, n_labels, cluster_label_mask=metric_fc_kwargs.pop("cluster_label_mask", None))
        elif metric == "cosine":
            self.metric_fc = Metric(hidden_dim, n_labels, rescale=metric_fc_kwargs.pop("rescale", None))
        elif metric == "clustered_cosine":
            self.metric_fc = Metric(hidden_dim, n_labels, rescale=metric_fc_kwargs.pop("rescale", None), cluster_label_mask=metric_fc_kwargs.pop("cluster_label_mask", None))
        else:
            raise Exception(f"Unknown metric {metric}")
        if loss == "cross_entropy":
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            raise Exception(f"Unknown loss {loss}")

        self.mask_and_shuffle = mask_and_shuffle
        self.batch_norm = torch.nn.BatchNorm1d(hidden_dim, momentum=batch_norm_momentum, affine=batch_norm_affine)

    def forward(self,
                tokens,
                mask,
                groups,
                labels=None,
                return_loss=False,
                return_scores=False,
                return_embeds=False):
        if self.mask_and_shuffle is not None and self.training:
            tokens, mask = compute_mask_and_shuffle(
                tokens,
                mask,
                min_size=self.mask_and_shuffle[0],
                dep_std=self.mask_and_shuffle[1],
                masking_rate=self.mask_and_shuffle[2],
                keep_first_and_last=True)

        # Embed the tokens
        scores = None
        # shape: n_batch * sequence * 768
        embeds = self.embeddings(tokens, mask)[0]
        state = embeds.masked_fill(~mask.unsqueeze(-1), 0)

        state[:, 0] = 0
        state[range(len(state)), mask.sum(-1) - 1] = 0
        state = state.sum(-2) / (mask.sum(-1) - 2).unsqueeze(-1)
        # shape state: n_batch * 768

        if return_loss or return_scores or return_embeds:
            state = F.normalize(self.batch_norm(torch.relu(self.linear(self.dropout(state)))), dim=-1)
        if return_loss or return_scores:
            scores = self.metric_fc(state, clusters=groups)  # n_batch * n_labels

        loss = None
        if return_loss:
            loss = self.loss(scores, labels)

        return {
            "loss": loss,
            "embeds": state,
            "scores": scores,
        }
