import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Optional,
    Union,
)

import edsnlp
import polars as pl
import spacy.tokens
import torch
import torch.utils.data
from accelerate.utils import send_to_device
from confit import Cli, validate_arguments
from confit.utils.random import set_seed
from edsnlp.core.pipeline import Pipeline
from edsnlp.optimization import LinearSchedule, ScheduledOptimizer
from edsnlp.pipes.trainable.span_linker.span_linker import TrainableSpanLinker
from edsnlp.scorers import make_examples
from edsnlp.scorers.span_classification import span_classification_scorer
from edsnlp.train import flatten_dict
from edsnlp.utils.bindings import BINDING_SETTERS
from edsnlp.utils.collections import batchify
from edsnlp.utils.span_getters import get_spans
from rich_logger import RichTablePrinter
from tqdm import tqdm, trange

from mlg_norm.slice_context import SliceContext

BASE_DIR = Path.cwd()

app = Cli(pretty_exceptions_show_locals=False)

LOGGER_FIELDS = {
    "step": {},
    "(.*)loss": {
        "goal": "lower_is_better",
        "format": "{:.2e}",
        "goal_wait": 2,
    },
    "micro/(f|p|r|ap)$": {
        "goal": "higher_is_better",
        "format": "{:.2%}",
        "goal_wait": 1,
        "name": r"\1",
    },
    "lr": {"format": "{:.2e}"},
    "labels": {"format": "{:.2f}"},
    "bias": {"format": "{:.2f}"},
}


@validate_arguments
def load_umls(
    path: Path,
    query: str,
    cui_query: Optional[str] = None,
    secondary_query: Optional[str] = None,
) -> pl.LazyFrame:
    # Load the terminology
    sty_groups = pl.scan_csv(
        path / "sty_groups.tsv",
        separator="|",
        new_columns="GRP|GRP_FULL|TUI|Full Semantic Type Name".split("|"),
        has_header=False,
        encoding="utf8-lossy",
    )
    mrsty = (
        pl.scan_csv(
            path / "MRSTY.RRF",
            separator="|",
            new_columns="CUI|TUI|STN|STY|ATUI|CVF|".split("|"),
            has_header=False,
            encoding="utf8-lossy",
        )
        .select("CUI", "TUI")
        .join(sty_groups, on="TUI")
    )
    mrconso = pl.scan_csv(
        path / "MRCONSO.RRF",
        separator="|",
        new_columns=(
            "CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|"
            "SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF|".split("|")
        ),
        has_header=False,
        encoding="utf8-lossy",
    )

    filtered = mrconso = mrconso.join(mrsty, on="CUI")
    subsets = []

    if query:
        filtered = filtered.filter(pl.sql_expr(query))

    if cui_query:
        non_eng_cuis = filtered.filter(pl.sql_expr(cui_query)).select("CUI").unique()
        filtered = filtered.join(non_eng_cuis, on="CUI")

    subsets.append(filtered.select("STR", "CUI", "GRP"))

    if secondary_query:
        filtered = mrconso.filter(pl.sql_expr(secondary_query))
        subsets.append(filtered.select("STR", "CUI", "GRP"))

    mrconso = pl.concat(subsets).with_columns(pl.col("STR").str.to_lowercase()).unique()
    mrconso = mrconso.sort(pl.col("STR").str.n_chars(), descending=True)

    return mrconso


def convert(entry, tokenizer=None):
    doc = tokenizer(entry["STR"].lower()[:100])
    span = spacy.tokens.Span(
        doc,
        0,
        len(doc),
        label=entry["GRP"],
    )
    span._.cui = entry["CUI"]
    doc.spans["entities"] = [span]
    return doc


def evaluate_model(nlp, docs):
    linker: TrainableSpanLinker = nlp.pipes.linker
    clean_docs = [d.copy() for d in docs]
    for doc in clean_docs:
        for span in get_spans(doc, linker.span_getter):
            BINDING_SETTERS[("_." + linker.attribute, None)](span)
    with nlp.select_pipes(enable=["linker"]):
        preds = list(nlp.pipe(tqdm(clean_docs, desc="Predicting")))
    return span_classification_scorer(
        make_examples(docs, preds),
        span_getter=linker.span_getter,
        qualifiers={linker.attribute: True},
    )


@app.command(name="pretrain")
def pretrain(
    *,
    nlp: Pipeline,
    seed: int = 42,
    max_steps: int = 20000,
    val_docs: Any,
    transformer_lr: float = 5e-5,
    task_lr: float = 1e-4,
    batch_size: int = 512,
    validation_interval: Optional[int] = None,
    max_grad_norm: float = 10.0,
    warmup_rate: float = 0.1,
    output_dir: Path = "artifacts/model-inter",
    umls_path: Path,
    query: str,
    cui_query: Optional[str] = None,
    cpu: bool = False,
    dropout: float = 0.2,
    debug: bool = False,
):
    """
    Pre-train an embedding model to link entities to concepts. This step is longer
    than the classifier training step but is meant to teach the embedding to:

    - learn multilingual representations (if any), following the paper's method
    - learn to project synonyms of the same concept close to each other

    Parameters
    ----------
    nlp : Pipeline
        The pipeline to train (expects a "linker" pipe)
    seed : int
        The seed for the random number generator.
    max_steps : int
        The maximum number of steps to train the model.
    val_docs : Any
        The validation documents.
    transformer_lr : float
        The learning rate for the transformer, set to 0 to freeze the transformer.
    task_lr : float
        The learning rate for the non-transformer parts of the model (e.g.
        the classifier).
    batch_size : int
        The batch size
    validation_interval : Optional[int]
        The interval at which to validate the model. Defaults, to `max_steps` // 10.
    max_grad_norm : float
        The maximum gradient norm, set to 0 to disable gradient clipping.
    warmup_rate : float
        The warmup rate for the transformer learning rate schedules.
    scorer : GenericScorer
        The scorer to use.
    output_dir : Path
        The output directory.
    umls_path : Path
        The path to the UMLS data. Expects the following files:

        - MRCONSO.RRF
        - MRSTY.RRF
        - sty_groups.tsv
    query : str
        The polars query to filter the UMLS synonyms.
    cui_query : Optional[str]
        The polars query to filter the CUIs in the retrieved synonyms.
    cpu : bool
        Whether to force the use of the CPU (can be useful for debugging).
    dropout : float
        The dropout rate to apply to the entire model (including the transformer).
    debug : bool
        Whether to run in debug mode (max_steps = 500, limit synonyms = 10000).
    """
    print("Start pre-training entity linker")

    def make_batches(prep, nlp, batch_size, device):
        while True:
            shuffled = list(prep)
            random.shuffle(shuffled)
            for batch in batchify(shuffled, batch_size):
                batch = nlp.collate(batch)
                batch = send_to_device(batch, device)
                yield batch

    output_dir = Path(output_dir)
    train_metrics_path = output_dir / "train_metrics.json"
    device = torch.device("cuda" if not cpu and torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Training linker")
    linker: TrainableSpanLinker = nlp.pipes.linker
    set_seed(seed)
    for module in linker.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout

    # ------ Prepare the data ------
    synonyms = load_umls(
        umls_path,
        query=query,
        cui_query=cui_query,
    )
    if debug:
        max_steps = 500
        synonyms = synonyms.limit(10000).collect().lazy()

    # Preprocessing training data
    print("Preprocessing data")
    nlp.to(device)
    nlp.post_init(
        edsnlp.data.from_polars(
            synonyms, converter=convert, tokenizer=nlp.tokenizer
        ).set_processing(backend="multiprocessing", show_progress=True, batch_size=1024)
    )
    nlp.train(True)

    prep = list(
        edsnlp.data.from_polars(synonyms, converter=convert, tokenizer=nlp.tokenizer)
        .set_processing(backend="multiprocessing", show_progress=True, batch_size=1024)
        .map(nlp.preprocess, kwargs=dict(compress=True, supervision=True))
    )
    random.shuffle(prep)

    print("Stats")
    print("- Groups:", len(linker.span_labels_to_idx))
    print("- Concepts:", len(linker.concepts_to_idx))
    print("- Synonyms:", len(prep))

    # Optimizer
    params = set(linker.parameters())
    trf_pipe = linker.embedding
    trf_params = params & set(trf_pipe.parameters() if trf_pipe else ())
    batch_iterator = iter(make_batches(prep, nlp, batch_size, device))

    optim = ScheduledOptimizer(
        torch.optim.AdamW(
            [
                {
                    "params": list(params - trf_params),
                    "lr": task_lr,
                    "schedules": LinearSchedule(
                        total_steps=max_steps,
                        warmup_rate=warmup_rate,
                        start_value=task_lr,
                    ),
                }
            ]
            + [
                {
                    "params": list(trf_params),
                    "lr": transformer_lr,
                    "schedules": LinearSchedule(
                        total_steps=max_steps,
                        warmup_rate=warmup_rate,
                        start_value=0,
                    ),
                },
            ][: 1 if transformer_lr else 0]
        )
    )
    all_params = set(nlp.parameters())
    grad_params = {p for group in optim.param_groups for p in group["params"]}
    print(
        "Optimizing:"
        + "".join(
            f"\n - {len(group['params'])} params "
            f"({sum(p.numel() for p in group['params'])} total)"
            for group in optim.param_groups
        )
    )
    print(f"Not optimizing {len(all_params - grad_params)} params")
    for param in all_params - grad_params:
        param.requires_grad_(False)

    cumulated_data = defaultdict(lambda: 0.0, count=0)
    all_metrics = []
    set_seed(seed)
    scaler = torch.cuda.amp.GradScaler()

    logger = RichTablePrinter(LOGGER_FIELDS, auto_refresh=False)
    logger.hijack_tqdm()

    # Training loop - step 1
    validation_interval = validation_interval or max_steps // 10
    for step in trange(
        max_steps + 1,
        desc="Training model",
        leave=True,
        mininterval=5.0,
    ):
        if (step % validation_interval) == 0:
            with torch.cuda.amp.autocast():
                all_metrics.append(
                    {
                        "step": step,
                        "lr": optim.param_groups[0]["lr"],
                        "rescale": float(linker.classifier.rescale or 1.0),
                        "bias": float(linker.classifier.bias or 0.0),
                        **cumulated_data,
                        **evaluate_model(nlp, val_docs),
                    }
                )
                cumulated_data.clear()
                logger.log_metrics(flatten_dict(all_metrics[-1]))
                nlp.to_disk(output_dir)
                train_metrics_path.write_text(json.dumps(all_metrics, indent=2))

        if step == max_steps:
            break

        optim.zero_grad()
        mini_batch = next(batch_iterator)
        mini_batch = send_to_device(mini_batch, device)
        loss = torch.zeros((), device=device)
        with nlp.cache(), torch.cuda.amp.autocast():
            res = dict(linker.module_forward(mini_batch["linker"]))
            if "loss" in res:
                loss += res["loss"]
            for key, value in res.items():
                if key.endswith("loss"):
                    cumulated_data[key] += float(value)
            if torch.isnan(loss):
                raise ValueError("NaN loss")

            scaler.scale(loss).backward()
            del loss, res, key, value, mini_batch

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(grad_params, max_grad_norm)
        scaler.step(optim)
        scaler.update()


# noinspection PyTypeHints
@app.command(name="train_classifier")
def train_classifier(
    *,
    nlp: Union[Path, Pipeline],
    seed: int = 42,
    max_steps: int = 10000,
    val_docs: Any,
    task_lr: float = 1e-4,
    batch_size: int = 4,
    training_top_k: int = 200,
    validation_interval: Optional[int] = None,
    warmup_rate: float = 0.1,
    umls_path: Path,
    query: str,
    cui_query: Optional[str] = None,
    cpu: bool = False,
    mode: str = None,
    output_dir: Optional[Path] = "artifacts/model-last",
    enable_features_caching: bool = False,
    debug: bool = False,
):
    """
    Teach the model the concepts weights by probing the transformer embeddings.
    In this step, the embedding is frozen and the classifier is trained to predict
    the concept of synonyms drawn from the UMLS (using the provided filter queries).
    This step is faster and can be used to quickly add concepts to the model, but won't
    improve the model's multilingual or semantic properties.
    The faster training is enabled by the fact that the transformer is frozen and the
    input embeddings and the hard negatives can be pre-computed.

    Parameters
    ----------
    nlp : Union[Path, Pipeline]
        The pipeline to train (expects a "linker" pipe)
    seed : int
        The seed for the random number generator.
    max_steps : int
        The maximum number of steps to train the model.
    val_docs : Any
        The validation documents.
    task_lr : float
        The learning rate for the non-transformer parts of the model.
    batch_size : int
        The batch size
    training_top_k : int
        The number of hard negatives to sample during training.
    validation_interval : Optional[int]
        The interval at which to validate the model. Defaults, to `max_steps` // 10.
    warmup_rate : float
        The warmup rate for the transformer learning rate schedules.
    scorer : GenericScorer
        The scorer to use.
    umls_path : Path
        The path to the UMLS data. Expects the following files:

        - MRCONSO.RRF
        - MRSTY.RRF
        - sty_groups.tsv
    query : str
        The polars query to filter the UMLS synonyms.
    cui_query : Optional[str]
        The polars query to filter the CUIs in the retrieved synonyms.
    cpu : bool
        Whether to force the use of the CPU (can be useful for debugging).
    mode : str
        The reference mode to use for the linker (will override the provided pipeline
        mode). If "synonyms", the linker will use the synonyms as references without
        fine-tuning them.
    output_dir : Optional[Path]
        The output directory.
    enable_features_caching : bool
        Whether to cache the pre-computed features (embeddings, labels, hard negatives)
        and reuse them for the next training runs.
    debug : bool
        Whether to run in debug mode (max_steps = 500, limit synonyms = 10000).
    """

    def make_batches(
        *tensors,
        batch_size,
        device,
        shuffle_loop=False,
        show_progress=False,
    ):
        """
        Generate batches of inputs to train the model.
        """
        N = len(tensors[0])
        while True:
            indices = torch.randperm(N) if shuffle_loop else torch.arange(N)
            for i in (trange if show_progress else range)(
                0, N - batch_size + 1 if shuffle_loop else N, batch_size
            ):
                batch_indices = indices[i : i + batch_size]
                batch = [t[batch_indices] for t in tensors]
                batch = send_to_device(batch, device)
                yield batch
            if not shuffle_loop:
                break

    output_dir = Path(output_dir)
    train_metrics_path = output_dir / "train_metrics.json"
    device = torch.device("cuda" if not cpu and torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if isinstance(nlp, Path):
        nlp = edsnlp.load(
            nlp,
            overrides={
                "components": {
                    "linker": {
                        **({"reference_mode": mode} if mode is not None else {}),
                    },
                }
            },
        )
    nlp.pipes.linker: TrainableSpanLinker
    print("Training linker")
    linker: TrainableSpanLinker = nlp.pipes.linker
    set_seed(seed)
    nlp.to(device).train(False)

    # ------ Prepare the data ------
    synonyms = load_umls(
        umls_path,
        query=query,
        cui_query=cui_query,
    )
    if debug:
        max_steps = 500
        synonyms = synonyms.limit(10000).collect().lazy()

    # Preprocessing training data
    print("Preprocessing data")

    if not os.path.exists("cached_features.pt") or not enable_features_caching:
        all_concepts, all_labels, all_embeds = linker.compute_init_features(
            edsnlp.data.from_polars(
                synonyms, converter=convert, tokenizer=nlp.tokenizer
            ).set_processing(
                backend="multiprocessing", show_progress=True, batch_size=4096
            ),
            with_embeddings=True,
        )
        labels_to_cuis = defaultdict(set)
        for label, cui in zip(all_labels, all_concepts):
            if cui is not None:
                labels_to_cuis[label].add(cui)
        linker.update_concepts(
            concepts=all_concepts,
            mapping=labels_to_cuis,
            embeds=all_embeds,
        )

        all_concepts_indices = torch.tensor(
            [linker.concepts_to_idx[c] for c in all_concepts]
        )
        all_labels_indices = torch.tensor(
            [linker.span_labels_to_idx[lab] for lab in all_labels]
        )
        all_hard_negatives = []

        # Pre-compute the hard negatives
        with torch.no_grad(), torch.cuda.amp.autocast():
            for embeds, labels, targets in make_batches(
                all_embeds,
                all_labels_indices,
                all_concepts_indices,
                batch_size=batch_size,
                device=device,
                shuffle_loop=False,
                show_progress=True,
            ):
                logits = linker.classifier(embeds, labels)
                n = logits.shape[0]
                logits[torch.arange(n, device=device), targets] = -float("inf")
                hard_negatives = logits.topk(dim=-1, k=training_top_k)[1]
                all_hard_negatives.append(hard_negatives)

            all_hard_negatives = torch.cat(all_hard_negatives, dim=0)

        if enable_features_caching:
            torch.save(
                (
                    all_embeds,
                    all_labels_indices,
                    all_concepts_indices,
                    all_hard_negatives,
                    all_concepts,
                    labels_to_cuis,
                ),
                "cached_features.pt",
            )
    else:
        (
            all_embeds,
            all_labels_indices,
            all_concepts_indices,
            all_hard_negatives,
            all_concepts,
            labels_to_cuis,
        ) = torch.load("cached_features.pt", map_location=device)
        linker.update_concepts(
            concepts=all_concepts,
            mapping=labels_to_cuis,
            embeds=all_embeds.to(device),
        )
    # Pre-compute the synonyms embeddings
    logger = RichTablePrinter(LOGGER_FIELDS, auto_refresh=False)
    logger.hijack_tqdm()
    print("Stats")
    print("- Groups:", len(linker.span_labels_to_idx))
    print("- Concepts:", len(linker.concepts_to_idx))
    print("- Synonyms:", len(all_embeds))

    batch_iterator = make_batches(
        all_embeds,
        all_labels_indices,
        all_concepts_indices,
        all_hard_negatives,
        batch_size=batch_size,
        shuffle_loop=True,
        device=device,
    )
    del all_embeds, all_labels_indices, all_concepts_indices, all_hard_negatives

    optim = ScheduledOptimizer(
        torch.optim.AdamW(
            [
                {
                    "params": [linker.classifier.weight],
                    "lr": task_lr,
                    "schedules": LinearSchedule(
                        total_steps=max(1, max_steps),
                        warmup_rate=warmup_rate,
                        start_value=task_lr,
                    ),
                }
            ]
        )
    )
    all_params = set(nlp.parameters())
    grad_params = {p for group in optim.param_groups for p in group["params"]}
    print(
        "Optimizing:"
        + "".join(
            f"\n - {len(group['params'])} params "
            f"({sum(p.numel() for p in group['params'])} total)"
            for group in optim.param_groups
        )
    )
    print(f"Not optimizing {len(all_params - grad_params)} params")
    for param in all_params - grad_params:
        param.requires_grad_(False)

    set_seed(seed)

    optim.initialize()
    cumulated_data = defaultdict(lambda: 0.0, count=0)
    all_metrics = []

    validation_interval = validation_interval or max_steps // 10
    for step in trange(
        max_steps + 1,
        desc="Training model",
        leave=True,
        mininterval=5.0,
    ):
        if (step % max(1, validation_interval)) == 0:
            with torch.cuda.amp.autocast():
                all_metrics.append(
                    {
                        "step": step,
                        "lr": optim.param_groups[0]["lr"],
                        "rescale": float(linker.classifier.rescale or 1.0),
                        "bias": float(linker.classifier.bias or 0.0),
                        **cumulated_data,
                        **evaluate_model(nlp, val_docs),
                    }
                )
            cumulated_data.clear()
            logger.log_metrics(flatten_dict(all_metrics[-1]))
            nlp.to_disk(output_dir)
            train_metrics_path.write_text(json.dumps(all_metrics, indent=2))

        if step == max_steps:
            break

        with SliceContext(optim) as slice_context:
            mini_batch = next(batch_iterator)
            mini_batch = send_to_device(mini_batch, device)
            span_embeds, span_labels, targets, hard_negatives = mini_batch

            if training_top_k:
                subset_concepts, subset_targets = torch.cat(
                    [
                        targets,
                        hard_negatives.view(-1),
                    ]
                ).unique(return_inverse=True)
                targets = subset_targets[: len(targets)]

                # Only optimize the weights for the subset concepts
                slice_context.slice_param_(
                    module=linker.classifier,
                    names={"weight": 0, "groups": 1},
                    indexer=subset_concepts,
                )
            optim.zero_grad()

            with torch.cuda.amp.autocast():
                logits = linker.classifier(span_embeds, span_labels)
                if linker.probability_mode == "sigmoid":
                    loss = (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            logits,
                            torch.nn.functional.one_hot(
                                targets, num_classes=linker.classifier.weight.shape[0]
                            ).float(),
                            reduction="sum",
                        )
                        / linker.classifier.weight.shape[0]
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(
                        logits,
                        targets,
                        reduction="sum",
                    )
                cumulated_data["loss"] += float(loss)

            loss.backward()
            del loss, mini_batch

            optim.step()

    return nlp


if __name__ == "__main__":
    app()
