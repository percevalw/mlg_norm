import gc
import logging
import math
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm as tqdm
from transformers import AdamW
from transformers import AutoModel

from nlstruct.collections import Batcher
from nlstruct.environment import get_cache, yaml_dump, cached
from nlstruct.train import seed_all, make_optimizer_and_schedules, run_optimization, fork_rng, iter_optimization
from nlstruct.train.schedule import LinearSchedule, CosineSchedule
from nlstruct.utils import evaluating, torch_global as tg, freeze, print_optimized_params, unfreeze, factorize
from nlstruct.utils import torch_clone
from .eval import compute_scores, predict
from .model import Classifier, FastClusteredIPSearch
from .utils import slice_parameters

# Define the training metrics
flt_format = (5, "{:.4f}".format)
metrics_info = {
    "step": {"format": (8, lambda x: str(int(x)))},
    "train_loss": {"goal": 0, "format": flt_format},
    "lr": {"format": (9, "{:.3e}".format)},
    "rescale": {"format": flt_format},
    "norm": {"format": (6, "{:.2f}".format)},
    "train_acc": {"goal": 1, "format": flt_format},
    "train_recall": {"goal": 1, "format": flt_format, "name": "train_rec"},
    "train_precision": {"goal": 1, "format": flt_format, "name": "train_prec"},
    "train_f1": {"goal": 1, "format": flt_format, "name": "train_f1"},
    "val_loss": {"goal": 0, "format": flt_format},
    "val_acc": {"goal": 1, "format": flt_format},
    "val_map": {"goal": 1, "format": flt_format},
    "val_acc_emea": {"goal": 1, "format": flt_format, "name": "acc_emea"},
    "val_acc_medline": {"goal": 1, "format": flt_format, "name": "acc_medline"},
    "val_acc_before": {"goal": 1, "format": flt_format},
    "val_recall": {"goal": 1, "format": flt_format, "name": "val_rec"},
    "val_precision": {"goal": 1, "format": flt_format, "name": "val_prec"},
    "val_f1": {"goal": 1, "format": flt_format, "name": "val_f1"},
    "duration": {"format": flt_format, "name": "   dur(s)"},
}


def clear():
    sys.last_value = None
    sys.last_traceback = None
    sys.last_type = None
    gc.collect()
    if tg.device != torch.device('cpu'):
        with torch.cuda.device(tg.device):
            torch.cuda.empty_cache()


if "BERTS" not in globals():
    BERTS = {}


def train_step1(
      # Data
      train_batcher,
      val_batcher,
      vocabularies,
      group_label_mask,

      # Learning rates
      metric_lr,
      inter_lr,
      bert_lr,

      # Misc params
      bert_name,
      metric,
      dim,
      rescale,
      batch_norm_affine,
      batch_norm_momentum,
      train_with_groups,

      # Regularizers
      dropout,
      bert_dropout,
      mask_and_shuffle,
      n_freeze,
      sort_noise,
      n_neighbors,

      # Scheduling
      batch_size,
      max_epoch,
      bert_warmup,
      decay_schedule,

      # Experiment params
      seed,
      bert_unfreeze_ratio=1,
      from_tf=True,
      stop_epoch=None,
      with_cache=True,
      debug=False,
      with_tqdm=False,

      device=None,
):
    if device is None:
        device = tg.device

    if debug:
        with_cache = False
        stop_epoch = 2

    # Load BERT
    if bert_name not in BERTS:
        BERTS.clear()
        with fork_rng(42):
            BERTS[bert_name] = AutoModel.from_pretrained(bert_name, from_tf=from_tf)

    seed_all(seed)  # /!\ Super important to enable reproducibility
    if stop_epoch is None:
        stop_epoch = max_epoch

    val_batcher_emea = val_batcher[val_batcher['quaero_source'] == 0]  # EMEA
    val_batcher_medline = val_batcher[val_batcher['quaero_source'] == 1]  # MEDLINE

    #######################
    # Models & optimizers #
    #######################
    classifier = Classifier(
        n_tokens=len(vocabularies["token"]),
        token_dim=1024 if "large" in bert_name else 768,
        n_labels=len(vocabularies['label']),

        ##############
        # EMBEDDINGS #
        ##############
        embeddings=torch_clone(BERTS[bert_name]),

        dropout=dropout,
        hidden_dim=dim,
        metric=metric,
        metric_fc_kwargs={"cluster_label_mask": torch.as_tensor(group_label_mask.toarray()), "rescale": rescale},
        loss='cross_entropy',
        mask_and_shuffle=mask_and_shuffle,

        batch_norm_affine=batch_norm_affine,
        batch_norm_momentum=batch_norm_momentum,
    ).train().to(device)

    # Freeze some bert layers
    # - n_freeze = 0 to freeze nothing
    # - n_freeze = 1 to freeze word embeddings / position embeddings / ...
    # - n_freeze = 2..13 to freeze the first, second ... last layer of bert
    for name, param in classifier.named_parameters():
        match = re.search("\.(\d+)\.", name)
        if match and int(match.group(1)) < n_freeze - 1:
            freeze([param])
    if n_freeze > 0:
        if hasattr(classifier.embeddings, 'embeddings'):
            freeze(classifier.embeddings.embeddings)
        else:
            freeze(classifier.embeddings)

    for module in classifier.embeddings.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = bert_dropout

    # Define the optimizer, maybe multiple learning rate / schedules per parameters groups
    if sort_noise is not None:
        sort_noise = float(sort_noise)
        dataloader = train_batcher.dataloader(batch_size=batch_size, shuffle=True, keys_noise=sort_noise, sort_on=["token_mask"], device=tg.device)
    else:
        dataloader = train_batcher.dataloader(batch_size=batch_size, shuffle=True, device=tg.device)

    total_steps = len(dataloader) * max_epoch
    decay_schedule_cls = LinearSchedule if decay_schedule == "linear" else CosineSchedule if decay_schedule == "cosine" else None
    optim, schedules = make_optimizer_and_schedules(classifier, AdamW, {
        # "scale_parameter": False,
        "lr": [x for x in (
            (inter_lr, 0, metric_lr) if bert_warmup > 0 else (inter_lr, bert_lr, metric_lr),
            (LinearSchedule, (inter_lr, bert_lr, metric_lr), math.ceil(total_steps * bert_warmup)),
            (decay_schedule_cls, (0, 0, 0), total_steps - math.ceil(total_steps * bert_warmup)) if decay_schedule_cls is not None else None,
        ) if x is not None],
    }, ["(?!embeddings\.|metric_fc).*", "embeddings\..*", "metric_fc\..*"], num_iter_per_epoch=1)
    # classifier.metric_fc.weight.data *= 2e-5 / 8e-3
    if n_neighbors is not None:
        classifier.metric_fc.weight.grad = torch.zeros_like(classifier.metric_fc.weight.data)
        optim.step()

    if debug:
        print_optimized_params(classifier, optim)

    # To debug the training, we can just comment the "def run_epoch()" and execute the function body manually without changing anything to it
    def run_epoch():
        #################
        # TRAINING STEP #
        #################
        total_train_tp = 0
        total_train_size = 0
        total_train_loss = 0
        bert_params = list(classifier.embeddings.parameters())

        bar = tqdm.tqdm(dataloader, leave=False, desc="Training epoch {}".format(training_state["epoch"]), disable=not with_tqdm, mininterval=10)
        for i, batch in enumerate(bar):
            optim.zero_grad()

            if debug and training_state["step"] > 5:
                break

            with memoryview(b'') if i % bert_unfreeze_ratio == 0 else freeze(bert_params):
                embeds = classifier.forward(
                    tokens=batch["token"],
                    mask=batch["token_mask"],
                    labels=batch["label"],
                    groups=batch["group"] if train_with_groups else None,
                    return_embeds=True,
                )["embeds"]

            if n_neighbors is not None:
                with torch.no_grad():
                    with evaluating(classifier.metric_fc):
                        scores = classifier.metric_fc(embeds, group=batch["group"] if train_with_groups else None)
                        neighbors = scores.topk(k=n_neighbors, dim=-1)[1].view(-1)
                        # scores = None
                        # neighbors = torch.randint(classifier.metric_fc.weight.shape[0], size=(6000,), device=device)
                [_1, targets], _2, label_subset = factorize([neighbors, batch["label"]])
                del neighbors, scores
            else:
                targets = batch["label"]
                label_subset = None

            to_slice = {'weight': 0, 'cluster_label_mask': 1, 'bias': 0}
            to_slice = {k: v for k, v in to_slice.items() if hasattr(classifier.metric_fc, k)}
            with (
                  slice_parameters(classifier.metric_fc, to_slice, label_subset, optim, device=device)
                  if n_neighbors is not None else memoryview(b'')
            ):
                scores = classifier.metric_fc(embeds, group=batch["group"])
                loss = F.cross_entropy(scores, targets)
                pred_targets = scores.argmax(-1)

                # Perform optimization step
                loss.backward()
                optim.step()

            for schedule_name, schedule in schedules.items():
                schedule.step()

            total_train_size += len(batch)
            total_train_loss += float(loss * len(batch))
            total_train_tp += float((pred_targets == targets).float().sum())

            training_state["step"] += 1

            bar.set_postfix(
                loss=float(total_train_loss / total_train_size),
                acc=float(total_train_tp / total_train_size),
                subset=len(label_subset) if label_subset is not None else None,
                lr=float(optim.param_groups[0]["lr"]),
                norm=float(classifier.metric_fc.weight.data.norm(dim=-1).mean()),
                refresh=False)
            # records.append({"loss": loss.item(), "lr": schedules["lr"].get_val()[-1]})

        val_metrics = compute_scores(predict(val_batcher, classifier), val_batcher)
        val_metrics_emea = compute_scores(predict(val_batcher_emea, classifier), val_batcher_emea)
        val_metrics_medline = compute_scores(predict(val_batcher_medline, classifier), val_batcher_medline)

        # Compute precision, recall and f1 on validation set
        return \
            {
                "train_loss": total_train_loss / max(total_train_size, 1),
                "train_acc": total_train_tp / max(total_train_size, 1),
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["f1"],
                "val_map": val_metrics["map"],
                "val_acc_emea": val_metrics_emea["f1"],
                "val_acc_medline": val_metrics_medline["f1"],
                "lr": float(optim.param_groups[0]["lr"]),
                "norm": float(classifier.metric_fc.weight.data.norm(dim=-1).mean()),
                "step": training_state["step"],
            }

    training_state = {
        "classifier": classifier,
        "optim": optim,
        "step": 0,
        **schedules,
    }
    hp = {
        "train_batcher": str(train_batcher),
        "val_batcher": str(val_batcher),
        "vocabularies": str({key: len(values) for key, values in vocabularies.items()}),
        "bert_name": bert_name,
        "metric_lr": metric_lr,
        "inter_lr": inter_lr,
        "bert_lr": bert_lr,
        "dim": dim,
        "rescale": rescale,
        "batch_norm_affine": batch_norm_affine,
        "batch_norm_momentum": batch_norm_momentum,
        "dropout": dropout,
        "bert_dropout": bert_dropout,
        "mask_and_shuffle": str(mask_and_shuffle),
        "n_freeze": n_freeze,
        "sort_noise": sort_noise,
        "batch_size": batch_size,
        "max_epoch": max_epoch,
        "bert_warmup": bert_warmup,
        "seed": seed,
        "train_with_groups": train_with_groups,
        **({"bert_unfreeze_ratio": bert_unfreeze_ratio} if bert_unfreeze_ratio != 1 else {}),
        **({"n_neighbors": n_neighbors} if n_neighbors is not None else {}),
    }
    if with_cache:
        cache = get_cache("norm/paper/train_step1", {
            **training_state,
            **hp,
            "train_batcher": train_batcher,
            "val_batcher": val_batcher,
        }, loader=torch.load, dumper=torch.save)
        cache.dump(hp, "hp.yaml", dumper=yaml_dump)
    else:
        cache = None

    best, history = run_optimization(
        metrics_info=metrics_info,
        max_epoch=stop_epoch,  # Max number of epochs

        state=training_state,
        seed=seed,

        cache=cache,
        epoch_fn=run_epoch,
    )
    history = pd.DataFrame(history)
    history["epoch"] = np.arange(len(history))
    history = history.assign(**hp)
    return history, classifier


def loader(path):
    return torch.load(path, map_location=tg.device)


@cached(loader=loader, dumper=torch.save)
def train_step2(
      classifier,

      train_batcher,
      val_batcher,
      group_label_mask,

      batch_size=128,
      sort_noise=1.,
      decay_schedule="constant",
      lr=8e-3,
      n_epochs=5,
      seed=42,
      init="mean",
      rescale=None,
      n_neighbors=100,
):
    assert init in ("mean", "random")
    clear()

    val_batcher_emea = val_batcher[val_batcher['quaero_source'] == 0]  # EMEA
    val_batcher_medline = val_batcher[val_batcher['quaero_source'] == 1]  # MEDLINE

    old_weight_norm = classifier.metric_fc.weight.norm(dim=-1).mean().item()
    classifier.metric_fc.weight = None
    classifier.metric_fc.cluster_label_mask = None

    #######################################################
    # Build & store in RAM every train synonym embeddings #
    #######################################################
    train_embeds, new_weights = predict(train_batcher, classifier, save_embeds=True, topk=0, batch_size=512, with_tqdm=True, sum_embeds_by_label=group_label_mask.shape[1], return_loss=False)
    train_embeds = train_embeds.merge(train_batcher["mention", ["mention_id", "label", "group"]])
    classifier.cpu()
    classifier.metric_fc.weight = torch.nn.Parameter(F.normalize(new_weights, dim=-1) * old_weight_norm, requires_grad=True)
    if init == "random":
        classifier.metric_fc.reset_parameters()

    classifier.metric_fc.cluster_label_mask = torch.as_tensor(group_label_mask.toarray())
    classifier.metric_fc.to(tg.device)

    torch.cuda.synchronize(tg.device)

    ##########################################################
    # Get the most probable concepts for every train synonym #
    ##########################################################
    neighbors = np.zeros((len(train_batcher), n_neighbors,), dtype=np.long)
    ids = np.zeros((len(train_batcher),), dtype=np.long)
    # index = FastClusteredIPSearch(dim=classifier.metric_fc.weight.shape[1], n_clusters=classifier.metric_fc.cluster_label_mask.shape[0], device=tg.device)
    # index.train(classifier.metric_fc.weight.data.clone(), cluster_label_mask=classifier.metric_fc.cluster_label_mask)

    i = 0
    with torch.no_grad():
        with evaluating(classifier):
            with tqdm.tqdm(train_embeds.dataloader(512, device=tg.device, sort_on="group", dtypes={"embed": torch.float32}), mininterval=10) as bar:
                for batch in bar:
                    closest = classifier.metric_fc(batch['embed'].to(torch.float32), clusters=batch['group']).topk(k=n_neighbors)[1]
                    missing_gold = ~(closest == batch['label'].unsqueeze(-1)).any(-1)
                    closest[missing_gold, -1] = batch['label'][missing_gold]
                    neighbors[i:i + len(batch)] = closest.cpu().numpy()
                    ids[i:i + len(batch)] = batch["mention_id"].cpu().numpy()
                    i += len(batch)
    del train_embeds
    clear()
    train_batcher = train_batcher.merge(Batcher({"mention": {"neighbors": neighbors, "mention_id": ids}}))
    del neighbors, ids

    ################################
    # Finetune the last layer only #
    ################################
    iterator = iter_optimization(metrics_info=metrics_info, state={}, max_epoch=n_epochs)
    iterator.score_logger.display({
        "epoch": 0,
        "val_acc": compute_scores(predict(val_batcher, classifier.to(tg.device), batch_size=32), val_batcher)["f1"],
        "val_loss": None,
        "val_acc_emea": compute_scores(predict(val_batcher_emea, classifier.to(tg.device), batch_size=32), val_batcher_emea)["f1"],
        "val_acc_medline": compute_scores(predict(val_batcher_medline, classifier.to(tg.device), batch_size=32), val_batcher_medline)["f1"],
        "train_loss": None,
        "rescale": None,
        "lr": None,
    })

    if rescale == "dynamic":
        classifier.metric_fc.rescale = torch.nn.Parameter(torch.as_tensor([float(classifier.metric_fc.rescale)], device=tg.device)[0], requires_grad=True)
    elif isinstance(rescale, (float, int)):
        classifier.metric_fc.rescale = torch.nn.Parameter(torch.as_tensor([float(rescale)], device=tg.device)[0], requires_grad=True)

    classifier.metric_fc.weight.data = classifier.metric_fc.weight.data.cpu()
    clear()
    seed_all(seed)
    dataloader = train_batcher.dataloader(batch_size=batch_size, device=tg.device, keys_noise=sort_noise, sort_on="token_mask", shuffle=True)
    freeze(list(classifier.parameters()))
    unfreeze([classifier.metric_fc.weight])
    if rescale == "dynamic":
        unfreeze([classifier.metric_fc.rescale])

    total_steps = len(dataloader) * n_epochs
    schedule_map = {
        "cosine": CosineSchedule,
        "linear": LinearSchedule,
    }
    optim, schedules = make_optimizer_and_schedules(classifier.metric_fc, AdamW, {
        "lr": [
                  (lr, lr),
              ] + ([] if decay_schedule == "constant" else [(schedule_map[decay_schedule], (0, 0), total_steps)]),
    }, ["rescale", "weight"], num_iter_per_epoch=1)

    for param in optim.param_groups[1]["params"]:
        print(param.shape, param.device)

    classifier.metric_fc.weight.grad = torch.zeros_like(classifier.metric_fc.weight)
    if isinstance(classifier.metric_fc.rescale, torch.nn.Parameter):
        classifier.metric_fc.rescale.grad = torch.zeros_like(classifier.metric_fc.rescale)
    optim.step()

    for epoch, state, history, record in iterator:
        classifier.metric_fc.weight.cpu()
        total_loss = 0
        total_size = 0
        with tqdm.tqdm(dataloader, leave=False, mininterval=10) as bar:
            for batch in bar:
                batch_closest_concepts = torch.unique(batch["neighbors"].view(-1))
                rel_labels = factorize(batch['label'], reference_values=batch_closest_concepts)[0]

                # Slice the concept matrix and only keep the aggregated neighbors of samples in batch
                # and send those concepts to the GPU
                # Then perform optim (loss -> gradient -> update) on the GPU and copy back the new weights & optim params to the CPU
                with slice_parameters(classifier.metric_fc, {'weight': 0, 'cluster_label_mask': 1}, batch_closest_concepts, optim, device=tg.device):

                    optim.zero_grad()
                    res = classifier(
                        tokens=batch["token"],
                        mask=batch["token_mask"],
                        labels=rel_labels,
                        groups=batch["group"],
                        return_loss=True)
                    loss = res["loss"]
                    loss.backward()
                    norm = float(classifier.metric_fc.weight.data.norm(dim=-1).mean())
                    optim.step()
                    # classifier.metric_fc.weight.data = F.normalize(classifier.metric_fc.weight.data, dim=-1) * norm
                    for schedule_name, schedule in schedules.items():
                        schedule.step()
                    total_loss += float(res["loss"]) * len(batch)
                    total_size += len(batch)
                    del res, loss, batch
                    bar.set_postfix(loss=total_loss / total_size, rescale=classifier.metric_fc.rescale.item(), lr=float(optim.param_groups[1]['lr']), refresh=False)

        clear()
        val_metrics = compute_scores(predict(val_batcher, classifier.to(tg.device), batch_size=32), val_batcher)
        record({
            "val_acc": val_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_acc_emea": compute_scores(predict(val_batcher_emea, classifier.to(tg.device), batch_size=32), val_batcher_emea)["f1"],
            "val_acc_medline": compute_scores(predict(val_batcher_medline, classifier.to(tg.device), batch_size=32), val_batcher_medline)["f1"],
            "train_loss": total_loss / total_size,
            "rescale": classifier.metric_fc.rescale.item(),
            "lr": float(optim.param_groups[1]['lr']),
            "norm": float(classifier.metric_fc.weight.data.norm(dim=-1).mean()),
        })
    return classifier


def train_big(
      # Data
      train_batcher,
      val_batcher,
      vocabularies,
      group_label_mask,

      # Learning rates
      metric_lr,
      inter_lr,
      bert_lr,

      # Misc params
      bert_name,
      dim,
      rescale,
      batch_norm_affine,
      batch_norm_momentum,
      train_with_groups,
      n_neighbors,

      # Regularizers
      dropout,
      bert_dropout,
      mask_and_shuffle,
      n_freeze,
      sort_noise,

      # Scheduling
      batch_size,
      total_steps,
      warmup_steps,
      decay_schedule,

      # Experiment params
      seed,
      debug=False,
      stop_epoch=None,
      with_cache=True,
      verbose=False,
      with_tqdm=False,

      device=None,
):
    if device is None:
        device = tg.device
    if debug:
        with_cache = False
        verbose = True
        stop_epoch = 2

    # Load BERT
    if bert_name not in BERTS:
        BERTS.clear()
        with fork_rng(42):
            BERTS[bert_name] = AutoModel.from_pretrained(bert_name, from_tf=True)

    seed_all(seed)  # /!\ Super important to enable reproducibility

    val_batcher_emea = val_batcher[val_batcher['quaero_source'] == 0]  # EMEA
    val_batcher_medline = val_batcher[val_batcher['quaero_source'] == 1]  # MEDLINE

    #######################
    # Models & optimizers #
    #######################
    classifier = Classifier(
        n_tokens=len(vocabularies["token"]),
        token_dim=1024 if "large" in bert_name else 768,
        n_labels=len(vocabularies['label']),

        ##############
        # EMBEDDINGS #
        ##############
        embeddings=torch_clone(BERTS[bert_name]),

        dropout=dropout,
        hidden_dim=dim,
        metric='clustered_cosine',
        metric_fc_kwargs={"cluster_label_mask": torch.as_tensor(group_label_mask.toarray()), "rescale": rescale},
        loss='cross_entropy',
        mask_and_shuffle=mask_and_shuffle,

        batch_norm_affine=batch_norm_affine,
        batch_norm_momentum=batch_norm_momentum,
    ).train().to(device)
    logging.info(f"Initial norm: {float(classifier.metric_fc.weight.norm(dim=-1).mean())}")

    # Freeze some bert layers
    # - n_freeze = 0 to freeze nothing
    # - n_freeze = 1 to freeze word embeddings / position embeddings / ...
    # - n_freeze = 2..13 to freeze the first, second ... last layer of bert
    for name, param in classifier.named_parameters():
        match = re.search("\.(\d+)\.", name)
        if match and int(match.group(1)) < n_freeze - 1:
            freeze([param])
    if n_freeze > 0:
        if hasattr(classifier.embeddings, 'embeddings'):
            freeze(classifier.embeddings.embeddings)
        else:
            freeze(classifier.embeddings)

    for module in classifier.embeddings.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = bert_dropout

    # Define the optimizer, maybe multiple learning rate / schedules per parameters groups
    if sort_noise is not None:
        sort_noise = float(sort_noise)
        dataloader = train_batcher.dataloader(batch_size=batch_size, shuffle=not debug, keys_noise=sort_noise, sort_on=["token_mask"], sort_keys="descending" if debug else "ascending",
                                              device=tg.device)
    else:
        dataloader = train_batcher.dataloader(batch_size=batch_size, shuffle=not debug, sort_keys="descending" if debug else "ascending", device=tg.device)

    # classifier.metric_fc.cpu()
    if stop_epoch is None:
        stop_epoch = math.ceil(total_steps / len(dataloader))
    decay_schedule_cls = {"linear": LinearSchedule, "cosine": CosineSchedule, "constant": None}[decay_schedule]
    optim, schedules = make_optimizer_and_schedules(classifier, AdamW, {
        "lr": [
                  (inter_lr, 0, metric_lr),
                  (LinearSchedule, (inter_lr, bert_lr, metric_lr), warmup_steps),
                  (decay_schedule_cls, (0, 0, 0), total_steps - warmup_steps),
              ][:(3 if decay_schedule_cls is not None else 2)],
    }, ["(?!embeddings\.|metric_fc).*", "embeddings\..*", "metric_fc\..*"], num_iter_per_epoch=1)
    classifier.metric_fc.weight.grad = torch.zeros_like(classifier.metric_fc.weight)
    optim.step()

    if verbose:
        print_optimized_params(classifier, optim)

    # To debug the training, we can just comment the "def run_epoch()" and execute the function body manually without changing anything to it
    def run_epoch():
        clear()
        # index.reset()
        # index.train(classifier.metric_fc.weight)
        # classifier.metric_fc.cpu()

        #################
        # TRAINING STEP #
        #################
        total_train_tp = 0
        total_train_size = 0
        total_train_loss = 0

        with tqdm.tqdm(dataloader, leave=False, desc="Training epoch {}".format(training_state["epoch"]), disable=not with_tqdm, mininterval=10) as bar:
            for batch in bar:
                if debug and training_state["step"] > 200:
                    break
                if debug:
                    print("Shape", batch["token"].shape)

                if training_state["step"] >= total_steps:
                    break
                training_state["step"] += 1

                optim.zero_grad()

                embeds = classifier.forward(
                    tokens=batch["token"],
                    mask=batch["token_mask"],
                    labels=batch["label"],
                    groups=batch["group"] if train_with_groups else None,
                    return_embeds=True,
                )["embeds"]

                if n_neighbors is not None:
                    with torch.no_grad():
                        with evaluating(classifier.metric_fc):
                            scores = classifier.metric_fc(embeds, group=batch["group"] if train_with_groups else None)
                            neighbors = scores.topk(k=n_neighbors, dim=-1)[1].view(-1)
                            # neighbors = torch.randint(classifier.metric_fc.weight.shape[0], size=(n_neighbors*len(batch),), device=device)
                    [_1, targets], _2, label_subset = factorize([neighbors, batch["label"]])
                    del neighbors
                else:
                    targets = batch["label"]
                    label_subset = None

                with (
                      slice_parameters(classifier.metric_fc, {'weight': 0, 'cluster_label_mask': 1}, label_subset, optim, device=device)
                      if n_neighbors is not None else memoryview(b'')
                ):
                    scores = classifier.metric_fc(embeds, group=batch["group"])
                    loss = F.cross_entropy(scores, targets)
                    pred_targets = scores.argmax(-1)

                    # Perform optimization step
                    loss.backward()
                    optim.step()

                for schedule_name, schedule in schedules.items():
                    schedule.step()

                total_train_size += len(batch)
                total_train_loss += float(loss * len(batch))
                total_train_tp += float((pred_targets == targets).float().sum())

                bar.set_postfix(
                    loss=float(total_train_loss / total_train_size),
                    acc=float(total_train_tp / total_train_size),
                    rescale=float(classifier.metric_fc.rescale),
                    lr=float(optim.param_groups[0]["lr"]),
                    subset=len(label_subset) if label_subset is not None else len(vocabularies['label']),
                    refresh=False)

                del scores, loss, pred_targets, targets, embeds, label_subset, batch

        val_acc = compute_scores(predict(val_batcher, classifier, batch_size=32), val_batcher)['f1']
        val_acc_emea = compute_scores(predict(val_batcher_emea, classifier, batch_size=32), val_batcher_emea)['f1']
        val_acc_medline = compute_scores(predict(val_batcher_medline, classifier, batch_size=32), val_batcher_medline)['f1']

        # Compute precision, recall and f1 on validation set
        return \
            {
                "train_loss": total_train_loss / max(total_train_size, 1),
                "train_acc": total_train_tp / max(total_train_size, 1),
                "val_acc": val_acc,
                "val_acc_emea": val_acc_emea,
                "val_acc_medline": val_acc_medline,
                "rescale": float(classifier.metric_fc.rescale),
                "lr": float(optim.param_groups[0]["lr"]),
                "norm": float(classifier.metric_fc.weight.norm(dim=-1).mean()),
                "step": training_state["step"],
            }

    training_state = {
        "classifier": classifier,
        "optim": optim,
        "step": 0,
        **schedules,
    }
    hp = {
        "train_batcher": str(train_batcher),
        "val_batcher": str(val_batcher),
        "vocabularies": str({key: len(values) for key, values in vocabularies.items()}),
        "bert_name": bert_name,
        "metric_lr": metric_lr,
        "inter_lr": inter_lr,
        "bert_lr": bert_lr,
        "dim": dim,
        "rescale": rescale,
        "batch_norm_affine": batch_norm_affine,
        "batch_norm_momentum": batch_norm_momentum,
        "dropout": dropout,
        "bert_dropout": bert_dropout,
        "mask_and_shuffle": str(mask_and_shuffle),
        "n_freeze": n_freeze,
        "sort_noise": sort_noise,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "train_with_groups": train_with_groups,
        "n_neighbors": n_neighbors,
    }
    if with_cache:
        cache = get_cache("norm/paper/train_big", {
            **training_state,
            **hp,
            "train_batcher": train_batcher,
            "val_batcher": val_batcher,
        }, loader=torch.load, dumper=torch.save)
        cache.dump(hp, "hp.yaml", dumper=yaml_dump)
    else:
        cache = None

    best, history = run_optimization(
        metrics_info=metrics_info,
        max_epoch=stop_epoch,  # Max number of epochs

        state=training_state,
        seed=seed,

        cache=cache,
        epoch_fn=run_epoch,
    )
    history = pd.DataFrame(history)
    history["epoch"] = np.arange(len(history))
    history = history.assign(**hp)
    return history, classifier