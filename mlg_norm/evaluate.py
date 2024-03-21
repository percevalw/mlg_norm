import json
from pathlib import Path
from typing import (
    Any,
    List,
)

import edsnlp
import spacy.tokens
import torch
from confit import Cli
from confit.utils.random import set_seed
from edsnlp import registry
from edsnlp.pipes.trainable.span_linker.span_linker import TrainableSpanLinker
from edsnlp.scorers import make_examples
from edsnlp.scorers.span_classification import span_classification_scorer
from edsnlp.utils.bindings import BINDING_SETTERS
from edsnlp.utils.span_getters import get_spans
from rich_logger import RichTablePrinter
from tqdm import tqdm

app = Cli(pretty_exceptions_show_locals=False)

BASE_DIR = Path(__file__).parent.parent


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


@app.command(name="evaluate", registry=registry)
def evaluate(
    *,
    data: Any,
    model_path: Path = BASE_DIR / "artifacts/model-last",
    data_seed: int = 42,
):
    nlp = edsnlp.load(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    logger = RichTablePrinter({".*": True})
    logger.hijack_tqdm()
    with torch.cuda.amp.autocast():
        for t in [0, 0.4, 0.5, 0.6, 0.7, 0.8]:
            print("THRESHOLD =", t)
            nlp.pipes.linker.threshold = t
            test_metrics_path = BASE_DIR / "artifacts/test_metrics.json"
            with set_seed(data_seed):
                val_docs: List[spacy.tokens.Doc] = list(data)
                scores = {
                    "threshold": t,
                    **evaluate_model(nlp, val_docs),
                }
            logger.log_metrics(
                {
                    "threshold": scores["threshold"],
                    "f1": scores["micro"]["f"],
                    "ap": scores["micro"]["ap"],
                }
            )
            print(json.dumps(scores, indent=2))
            test_metrics_path.write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    app()
