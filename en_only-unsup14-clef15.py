import string
import logging
import sys

from nlstruct.utils import torch_clone
from nlstruct.utils import torch_global as tg
from deep_multilingual_normalization.preprocess import preprocess, load_quaero
from deep_multilingual_normalization.train import train_step1, train_step2, clear
from deep_multilingual_normalization.eval import compute_scores, predict

logging.basicConfig(level=logging.INFO, format="", stream=sys.stdout)
logging.getLogger("transformers").setLevel(logging.WARNING)

subs = [
    (r"(?<=[{}\\])(?![ ])".format(string.punctuation), r" "),
    (r"(?<![ ])(?=[{}\\])".format(string.punctuation), r" "),
    ("(?<=[a-zA-Z])(?=[0-9])", r" "),
    ("(?<=[0-9])(?=[A-Za-z])", r" "),
    ("[ ]{2,}", " ")
]

bert_name = "bert-base-multilingual-uncased"
tg.set_device('cpu')
device = tg.device

# Step 1
train_batcher, vocabularies, train_mentions, train_mention_ids, group_label_mask, quaero_batcher, quaero_mentions, quaero_mention_ids = preprocess(
    bert_name=bert_name,
    umls_versions=["2014AB"],
    source_full_lexicon=True,
    source_lat=["FRE"],
    add_quaero_splits=["train"],
    other_full_lexicon=False,
    other_lat=["ENG"],
    other_additional_labels=None,
    other_mirror_existing_labels=True,
    sty_groups=['ANAT', 'CHEM', 'DEVI', 'DISO', 'GEOG', 'LIVB', 'OBJC', 'PHEN', 'PHYS', 'PROC'],
    other_sabs=None,
    subs=subs,
    apply_unidecode=True,
    max_length=100,
)

history, classifier = train_step1(
    # Data
    train_batcher=train_batcher,
    val_batcher=quaero_batcher[quaero_batcher['split'] == 0],
    vocabularies=vocabularies,
    group_label_mask=group_label_mask,

    # Learning rates
    metric_lr=8e-3,
    inter_lr=8e-3,
    bert_lr=2e-5,

    # Misc params
    metric='clustered_cosine',
    dim=350,
    rescale=20,
    bert_name=bert_name,
    batch_norm_affine=True,
    batch_norm_momentum=0.1,
    train_with_groups=True,

    # Regularizers
    dropout=0.2,
    bert_dropout=0.2,
    mask_and_shuffle=(2, 0.5, 0.1),
    n_freeze=0,
    sort_noise=1.,
    n_neighbors=None,

    # Scheduling
    batch_size=128,
    bert_warmup=0.1,
    max_epoch=15,
    decay_schedule="linear",

    # Experiment params
    seed=123456,
    stop_epoch=None,
    with_cache=True,
    debug=False,
    from_tf=False,
    with_tqdm=True,
)

# Step 2
train_batcher, vocabularies, train_mentions, train_mention_ids, group_label_mask, quaero_batcher, quaero_mentions, quaero_mention_ids = preprocess(
    bert_name=bert_name,
    umls_versions=["2014AB"],
    source_full_lexicon=True,
    source_lat=["FRE"],
    add_quaero_splits=["train"],
    other_full_lexicon=True,
    other_lat=["ENG"],
    other_additional_labels=None,
    other_mirror_existing_labels=True,
    sty_groups=['ANAT', 'CHEM', 'DEVI', 'DISO', 'GEOG', 'LIVB', 'OBJC', 'PHEN', 'PHYS', 'PROC'],
    other_sabs=["CHV", "SNOMEDCT_US", "MTH", "NCI", "MSH"],
    subs=subs,
    apply_unidecode=True,
    max_length=100,
)

classifier.cpu()
classifier2 = train_step2(
    classifier=torch_clone(classifier).to(tg.device),

    train_batcher=train_batcher,
    val_batcher=quaero_batcher[quaero_batcher['split'] == 0],
    group_label_mask=group_label_mask,

    batch_size=128,
    sort_noise=1.,
    decay_schedule="linear",
    lr=8e-3,
    n_epochs=5,
    seed=123456,
    rescale=20,
    n_neighbors=100,
)