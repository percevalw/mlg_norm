import numpy as np
import pandas as pd
import tqdm

from nlstruct.collections import Batcher
from nlstruct.scoring import compute_metrics, merge_pred_and_gold
from nlstruct.utils import evaluating, torch, torch_global as tg


def predict(batcher, classifier, batch_size=128, topk=1, save_embeds=False, with_tqdm=False, sum_embeds_by_label=False, return_loss=True, with_groups=True, device=None):
    if device is None:
        device = tg.device
    classifier.to(device)

    embeds = None
    ids = None
    top_labels = None
    top_probs = None
    new_weights = None
    index = None
    loss = None

    i = 0
    with evaluating(classifier):  # eval mode: no dropout, frozen batch norm, etc
        with torch.no_grad():  # no gradients -> faster
            with tqdm.tqdm(batcher.dataloader(batch_size=batch_size, device=device, sparse_sort_on="token_mask"), disable=not with_tqdm, mininterval=10.) as bar:
                for batch in bar:
                    res = classifier.forward(
                        tokens=batch["token"],
                        mask=batch["token_mask"],
                        groups=batch["group"] if with_groups else None,
                        return_scores=topk > 0,
                        return_embeds=save_embeds,
                    )
                    # Store predicted mention ids since we sort mentions by size
                    if ids is None:
                        ids = np.zeros((len(batcher),), dtype=int)
                    ids[i:i + len(batch)] = batch["mention_id"].cpu().numpy()

                    # If we want to predict new weights, directly sum mention embeddings on the device
                    if sum_embeds_by_label is not False:
                        if new_weights is None:
                            new_weights = torch.zeros(sum_embeds_by_label, res["embeds"].shape[1], device=device)
                        new_weights.index_add_(0, batch["label"], res["embeds"])

                    if return_loss:
                        if loss is None:
                            loss = np.zeros(len(batcher), dtype=float)
                        targets = batch["label"].clone()
                        targets[targets == -1] = 0
                        batch_losses = - res["scores"].log_softmax(dim=-1)[range(len(batch)), targets].cpu().numpy()
                        batch_losses[batch["label"].cpu().numpy() == -1] = -1
                        loss[i:i + len(batch)] = batch_losses
                        if index is None:
                            index = np.full(len(batcher), fill_value=-1, dtype=float)
                        index[i:i + len(batch)] = res["scores"].argsort(-1, descending=True).argsort(-1)[range(len(batch)), targets].cpu().numpy()

                    # If inference time, store topk prediction and associated probabilities
                    if topk > 0:
                        if top_labels is None:
                            top_labels = np.zeros((len(batcher), topk), dtype=int)
                            top_probs = np.zeros((len(batcher), topk), dtype=float)
                        probs = res["scores"].softmax(-1)
                        best_probs, best_labels = probs.topk(dim=-1, k=topk)
                        top_labels[i:i + len(batch)] = best_labels.cpu().numpy()
                        top_probs[i:i + len(batch)] = best_probs.cpu().numpy()

                    # If we want to save embeddings
                    if save_embeds:
                        if embeds is None:
                            embeds = np.zeros((len(batcher), res["embeds"].shape[1]), dtype=float)
                        embeds[i:i + len(batch)] = res["embeds"].cpu().numpy()
                    i += len(batch)
    if return_loss:
        index[loss == -1] = np.inf
    pred = Batcher({
        "mention": {k: v for k, v in {
            "mention_id": ids,
            "embed": embeds,
            "label": top_labels,
            "prob": top_probs,
            "loss": loss,
            "index": index,
        }.items() if v is not None}}, check=False)

    if new_weights is not None:
        return pred, new_weights
    return pred


def compute_scores(pred_batcher, gold_batcher, prefix="", threshold=0):
    pred = pd.DataFrame({
        "mention_id": pred_batcher["mention_id"],
        "label": pred_batcher["label"][:, 0],
        "prob": pred_batcher['prob'][:, 0],
        "loss": pred_batcher['loss'],
    })
    loss = float(pred["loss"][(pred["loss"] >= 0) & (pred["loss"] < 10000)].mean())
    pred = pred[pred['prob'] >= threshold]
    gold = pd.DataFrame({
        "mention_id": gold_batcher["mention_id"],
        "label": gold_batcher["label"],
    })
    return {
        **compute_metrics(
            merge_pred_and_gold(pred, gold, on=["mention_id", "label"],
                                atom_pred_level=["mention_id"],
                                atom_gold_level=["mention_id"]),
            prefix=prefix),
        prefix + "loss": loss,
        prefix + "map": (1 / (pred_batcher["index"] + 1)).mean(),
    }