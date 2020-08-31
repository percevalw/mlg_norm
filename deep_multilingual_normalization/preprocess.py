import logging

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from nlstruct.collections import Batcher
from nlstruct.dataloaders import load_quaero
from nlstruct.environment import cached, root
from nlstruct.text import apply_substitutions, huggingface_tokenize
from nlstruct.utils import normalize_vocabularies, df_to_csr, encode_ids


@cached
def preprocess_training_data(
      bert_name,
      umls_versions=["2014AB"],
      add_quaero_splits=None,
      n_repeat_quaero_corpus=1,
      source_lat=["FRE"],
      source_full_lexicon=False,
      other_lat=["ENG"],
      other_full_lexicon=False,
      other_sabs=None,
      other_additional_labels=None,
      other_mirror_existing_labels=True,
      sty_groups=['ANAT', 'CHEM', 'DEVI', 'DISO', 'GEOG', 'LIVB', 'OBJC', 'PHEN', 'PHYS', 'PROC'],
      filter_before_preprocess=None,
      subs=None,
      max_length=100,
      apply_unidecode=False,
      prepend_labels=[],
      mentions=None,
      vocabularies=None,
      return_raw_mentions=False,
):
    """
    Build and preprocess the training data

    Parameters
    ----------
        bert_name: str
            Name of the bert tokenizer
        umls_versions: list of str
            UMLS versions (ex ["2014AB"])
        add_quaero_splits: list of str
            Add the mentions from these quaero splits
        n_repeat_quaero_corpus: int
            Number of time the quaero corpus mentions should be repeated
        source_lat: list of str
            Source languages
        source_full_lexicon: bool
            Add all french umls synonyms
        other_lat: list of str
            Other languages
        other_full_lexicon: bool
            Add all english umls synonyms
        other_sabs: list of str
            Filter only by these sources when querying the english umls
        other_additional_labels: list of str
            Query those additional labels in the english umls
        other_mirror_existing_labels: bool
            Query previously added concepts (french, quaero, etc) in the english umls
        sty_groups: list of str
            If given, filter the lexicons to only keep concepts that are in those groups
        filter_before_preprocess: str
            Apply a final filtering before deduplicating the mentions and preprocessing them
        subs: list of (str, str)
            Substitutions to perform on mentions
        apply_unidecode: bool
            Apply unidecode module on mentions
        max_length: int
            Cut mentions that are longer than this number of tokens (not wordpieces)
        prepend_labels: list of str
            Add these virtual (?) labels at the beginning of the vocabulary
        mentions: pd.DataFrame
            Clean/tokenize these mentions instead of the ones built using this function parameters
        vocabularies: dict
            Base vocabularies if any
        return_raw_mentions: bool
            Return the untokenized, raw selected mentions only

    Returns
    -------
    Batcher, Dict[str; np.ndarray], transformers.Tokenizer, pd.DataFrame, pd.DataFrame
        Training batcher,
        Vocabularies
        Huggingface tokenizer
        Raw mentions
        Mention ids to unique coded id mapping

    """

    if mentions is None:
        assert not (other_full_lexicon and other_sabs is None and (other_mirror_existing_labels or other_additional_labels is not None))

        mrconso = None
        if any([source_full_lexicon, other_full_lexicon, other_additional_labels, other_mirror_existing_labels]):
            sty_groups_mapping = pd.read_csv(root.resource(f"umls/sty_groups.tsv"), sep='|', header=None, index_col=False, names=["group", "sty"])
            if sty_groups is not None:
                sty_groups_mapping = sty_groups_mapping[sty_groups_mapping.group.isin(sty_groups)]
            ### Load the UMLS extract
            # MRCONSO
            logging.info("Loading MRCONSO...")
            mrconso = []
            for version in umls_versions:
                mrconso_version = pd.read_csv(
                    root.resource(f"umls/{version}/MRCONSO.RRF"),
                    sep="|",
                    header=None,
                    index_col=False,
                    names=["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"],
                    usecols=["CUI", "STR", "LAT", "SAB"],
                ).rename({"CUI": "label", "STR": "text"}, axis=1)
                mrsty = pd.read_csv(
                    root.resource(f"umls/{version}/MRSTY.RRF"),
                    sep="|",
                    header=None,
                    index_col=False,
                    names=["CUI", "TUI", "STN", "STY", "ATUI", "CVF"],
                    usecols=["CUI", "STY"],
                ).rename({"CUI": "label", "STY": "sty"}, axis=1)
                mrsty = mrsty.merge(sty_groups_mapping, on="sty")
                mrsty = mrsty.drop_duplicates(['label', 'group'])
                mrconso_version = mrconso_version.merge(mrsty)
                mrconso.append(mrconso_version)
                del mrconso_version, mrsty
            mrconso = pd.concat(mrconso)
            logging.info("Deduplicating MRCONSO...")
            mrconso = mrconso[~pd.isna(mrconso[["label", "text", "group", "LAT"]]).any(axis=1)]  # drop entries containing NaNs
            mrconso = mrconso.drop_duplicates().reset_index(drop=True)
            mrconso["mention_id"] = mrconso["label"] + "-" + mrconso["group"] + "-" + pd.Series(np.arange(len(mrconso))).astype(str)
            mrconso["split"] = "train"
            mrconso["source"] = mrconso["SAB"] + "-" + mrconso["LAT"]

        mentions = pd.DataFrame({"mention_id": [], "text": [], "label": [], "group": [], "source": [], "split": []}, dtype=str)

        #######################
        # UMLS MRCONSO FRENCH #
        #######################
        if source_full_lexicon:
            source_lexicon = mrconso.query('LAT.isin({})'.format(source_lat))
            source_lexicon = source_lexicon[["mention_id", "text", "label", "group", "split", "source"]]
            logging.info(f"French synonyms: {len(source_lexicon)}")
            logging.info(f"French labels: {len(source_lexicon.label.drop_duplicates())}")
            mentions = pd.concat([mentions, source_lexicon])
            del source_lexicon
        else:
            logging.info("No french synonym")

        ###############
        # LOAD QUAERO #
        ###############
        if add_quaero_splits is not None:
            assert isinstance(add_quaero_splits, list) and all(isinstance(s, str) for s in add_quaero_splits)

            quaero_docs, quaero_mentions, quaero_fragments, quaero_labels = load_quaero()[["docs", "mentions", "fragments", "labels"]]
            quaero_mentions = quaero_mentions.merge(quaero_fragments).drop(columns=["text"]).rename({"label": "group"}, axis=1).merge(quaero_docs, on=["doc_id"])
            quaero_mentions["text"] = quaero_mentions.apply(lambda row: row["text"][row["begin"]:row["end"]], axis=1)
            quaero_mentions = quaero_mentions.groupby(["doc_id", "mention_id"], as_index=False).agg({"text": " ".join, "split": "first", "group": "first"})
            quaero_mentions = quaero_mentions.merge(quaero_labels, on=["doc_id", "mention_id"]).rename({"cui": "label"}, axis=1)
            quaero_mentions["mention_id"] = quaero_mentions["doc_id"] + "-" + quaero_mentions["mention_id"] + "-" + quaero_mentions["cui_id"].astype(str)

            quaero_mentions = quaero_mentions[["mention_id", "text", "label", "group", "split"]].assign(source="real").query(f"split.isin({add_quaero_splits})")
            mentions = pd.concat([
                mentions,
                quaero_mentions,
            ])
            logging.info(f"Quaero mentions: {len(quaero_mentions)}")
            del quaero_docs, quaero_mentions, quaero_fragments, quaero_labels

        if other_full_lexicon or other_mirror_existing_labels or other_additional_labels:
            if other_lat is not None:
                other_lexicon = mrconso.query("LAT.isin({})".format(other_lat))
            else:
                other_lexicon = mrconso
            # we add english concepts of labels in quaero_test that are not necessarily in the training
            # otherwise we just add english concepts of labels that are already in train_docs
            if other_mirror_existing_labels or other_additional_labels is not None:
                other_labels = set()
                if other_mirror_existing_labels:
                    other_labels = set(mentions['label'])
                    logging.info(f"Mirrored labels: {len(other_labels)}")
                if other_additional_labels is not None:
                    other_labels |= set(other_additional_labels)
                    logging.info(f"Added additional labels in english lexicon: {len(set(other_additional_labels))}")
                logging.info(f"Queried english labels: {len(other_labels)}")
                mentions = pd.concat([
                    mentions,
                    other_lexicon[other_lexicon['label'].isin(other_labels)][["mention_id", "text", "label", "group", "split", "source"]],
                ])
                del other_labels
            if other_full_lexicon:
                if other_sabs is not None:
                    logging.info(f"Adding all english concepts from SABs: {other_sabs}")
                    mentions = pd.concat([
                        mentions,
                        other_lexicon[other_lexicon.SAB.isin(other_sabs)][["mention_id", "text", "label", "group", "split", "source"]],
                    ])
                else:
                    mentions = pd.concat([
                        mentions,
                        other_lexicon[["mention_id", "text", "label", "group", "split", "source"]],
                    ])
            del other_lexicon
        else:
            logging.info("No english synonym")
        del mrconso

        if filter_before_preprocess is not None:
            mentions = mentions.query(filter_before_preprocess)

        mentions = mentions.drop_duplicates("mention_id").sort_values("mention_id")
        mentions = pd.concat([
            mentions.query("source != 'real'").drop_duplicates(["label", "text", "group"]),
            mentions.query("source == 'real'"),
        ])
        logging.info(f"Total deduplicated synonyms: {len(mentions)}")
        logging.info(f"Total deduplicated labels: {len(mentions.label.drop_duplicates())}")

    if return_raw_mentions:
        return mentions
    raw_mentions, mentions = mentions, mentions.copy()

    ###############################
    # CLEAN AND TOKENIZE MENTIONS #
    ###############################
    # Clean the text / perform substitutions
    # `deltas` contains the character span shifts made by the substitutions
    # we will reuse it at the end of the notebook to convert map predictions on input text
    mentions = mentions.copy()
    mentions["text"] = mentions["text"].str.lower()
    mentions["mention_id"] = mentions["mention_id"].astype(str)
    if subs is not None:
        mentions = apply_substitutions(mentions, *zip(*subs), apply_unidecode=apply_unidecode, return_deltas=False, with_tqdm=True)
    mentions = pd.concat([
        mentions.query("source == 'real'")[["mention_id", "text", "label", "group", "source", "split"]],
        mentions.query("source != 'real'").drop_duplicates(["label", "text", "group"])[["mention_id", "text", "label", "group", "source", "split"]],
    ])

    if add_quaero_splits is not None and n_repeat_quaero_corpus > 1:
        real_mentions = mentions.query("source == 'real'")[["mention_id", "text", "label", "group", "source", "split"]]
        mentions = pd.concat([
            pd.concat([real_mentions.assign(mention_id=real_mentions.astype(str) + f'-{i}') for i in range(n_repeat_quaero_corpus)]),
            mentions.query("source != 'real'").drop_duplicates(["label", "text", "group"])[["mention_id", "text", "label", "group", "source", "split"]],
        ])
    else:
        mentions = pd.concat([
            mentions.query("source == 'real'")[["mention_id", "text", "label", "group", "source", "split"]],
            mentions.query("source != 'real'").drop_duplicates(["label", "text", "group"])[["mention_id", "text", "label", "group", "source", "split"]],
        ])

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    tokens = huggingface_tokenize(mentions, tokenizer, max_length=max_length, with_token_spans=False, unidecode_first=True, with_tqdm=True)

    # Construct charset from token->token_norm
    if vocabularies is None:
        vocabularies = {}
    [mentions, token], vocabularies = normalize_vocabularies(
        [mentions, tokens],
        vocabularies={
            "label": [*prepend_labels],
            "split": ["train", "dev", "test"],
            "quaero_source": ["EMEA", "MEDLINE"],
            **vocabularies},
        train_vocabularies={**{key: False for key in vocabularies}, "text": False, "label": True, }, verbose=1)

    group_to_label = mentions[["group", "label"]].drop_duplicates()
    group_to_label = df_to_csr(group_to_label["group"], group_to_label["label"])

    ######################
    # CREATE THE BATCHER #
    ######################
    mention_ids = encode_ids([mentions, tokens], "mention_id")
    token_ids = encode_ids([tokens], ("mention_id", "token_id"))

    train_batcher = Batcher({
        "mention": {
            "mention_id": mentions["mention_id"],
            "token": df_to_csr(tokens["mention_id"], tokens["token_idx"], tokens["token"].cat.codes, n_rows=len(mention_ids)),
            "token_mask": df_to_csr(tokens["mention_id"], tokens["token_idx"], n_rows=len(mention_ids)),
            "split": mentions["split"].cat.codes,
            "group": mentions["group"].cat.codes,
            "label": mentions["label"].cat.codes,
            "source": mentions["source"].cat.codes,
        }
    }, masks={"mention": {"token": "token_mask"}})

    return train_batcher, vocabularies, raw_mentions, mention_ids, group_to_label


def preprocess_quaero(
      bert_name,
      subs=None,
      apply_unidecode=False,
      max_length=100,
      return_raw_mentions=False,
      vocabularies=None,
):
    """
    Build and preprocess quaero

    Parameters
    ----------
    bert_name: str
        Name of the bert tokenizer
    subs: list of (str, str)
        Substitutions to perform on mentions
    apply_unidecode: bool
        Apply unidecode module on mentions
    max_length: int
        Cut mentions that are longer than this number of tokens (not wordpieces)
    return_raw_mentions: bool
        Return the untokenized, raw selected mentions only
    vocabularies: Dict[str; np.ndarray]
        Vocabularies

    Returns
    -------
    Batcher, Dict[str; np.ndarray], pd.DataFrame, pd.DataFrame
        Training batcher,
        Vocabularies
        Huggingface tokenizer
        Raw mentions
        Mention ids to unique coded id mapping
    """

    quaero_docs, quaero_mentions, quaero_fragments, quaero_labels = load_quaero()[["docs", "mentions", "fragments", "labels"]]
    quaero_mentions = quaero_mentions.merge(quaero_fragments).drop(columns=["text"]).rename({"label": "group"}, axis=1).merge(quaero_docs, on=["doc_id"])
    quaero_mentions["text"] = quaero_mentions.apply(lambda row: row["text"][row["begin"]:row["end"]], axis=1)
    quaero_mentions = quaero_mentions.groupby(["doc_id", "mention_id"], as_index=False).agg({"text": " ".join, "split": "first", "source": "first", "group": "first"})
    quaero_mentions = quaero_mentions.merge(quaero_labels, on=["doc_id", "mention_id"]).rename({"cui": "label", "source": "quaero_source"}, axis=1)

    mentions = quaero_mentions[["doc_id", "mention_id", "cui_id", "text", "label", "split", "quaero_source", "group"]].assign(source="real")
    logging.info(f"Quaero mentions: {len(mentions)}")
    del quaero_docs, quaero_mentions, quaero_fragments, quaero_labels

    if return_raw_mentions:
        return mentions
    raw_mentions = mentions.copy()

    ###############################
    # CLEAN AND TOKENIZE MENTIONS #
    ###############################
    # Clean the text / perform substitutions
    # `deltas` contains the character span shifts made by the substitutions
    # we will reuse it at the end of the notebook to convert map predictions on input text
    mentions = mentions.copy()
    mentions["text"] = mentions["text"].str.lower()
    if subs is not None:
        mentions = apply_substitutions(mentions, *zip(*subs), apply_unidecode=apply_unidecode, return_deltas=False)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    tokens = huggingface_tokenize(mentions[["doc_id", "mention_id", "cui_id", "text"]], tokenizer, max_length=max_length, with_token_spans=False, unidecode_first=True)

    # Construct charset from token->token_norm
    if vocabularies is None:
        vocabularies = {}
    [mentions, token], vocabularies = normalize_vocabularies(
        [mentions, tokens],
        vocabularies={"split": ["train", "dev", "test"], **vocabularies},
        train_vocabularies={"text": False}, verbose=1)

    ######################
    # CREATE THE BATCHER #
    ######################
    mention_ids = encode_ids([mentions, tokens], ("doc_id", "cui_id", "mention_id"))
    token_ids = encode_ids([tokens], ("doc_id", "cui_id", "mention_id", "token_id"))

    batcher = Batcher({
        "mention": {
            "mention_id": mentions["mention_id"],
            "token": df_to_csr(tokens["mention_id"], tokens["token_idx"], tokens["token"].cat.codes, n_rows=len(mention_ids)),
            "token_mask": df_to_csr(tokens["mention_id"], tokens["token_idx"], n_rows=len(mention_ids)),
            "split": mentions["split"].cat.codes,
            "group": mentions["group"].cat.codes,
            "quaero_source": mentions["quaero_source"].cat.codes,
            "label": mentions["label"].cat.codes,
            "source": mentions["source"].cat.codes,
        }
    }, masks={"mention": {"token": "token_mask"}})

    return batcher, vocabularies, raw_mentions, mention_ids


def preprocess(
      bert_name,
      umls_versions=["2014AB"],
      add_quaero_splits=None,
      n_repeat_quaero_corpus=1,
      source_lat=["FRE"],
      source_full_lexicon=False,
      other_lat=["ENG"],
      other_full_lexicon=False,
      other_sabs=None,
      other_additional_labels=None,
      other_mirror_existing_labels=True,
      sty_groups=['ANAT', 'CHEM', 'DEVI', 'DISO', 'GEOG', 'LIVB', 'OBJC', 'PHEN', 'PHYS', 'PROC'],
      filter_before_preprocess=None,
      subs=None,
      max_length=100,
      apply_unidecode=False,
      prepend_labels=[],
      mentions=None,
      return_raw_mentions=False,
      vocabularies=None,
):
    res = preprocess_training_data(
        bert_name=bert_name,
        umls_versions=umls_versions,
        add_quaero_splits=add_quaero_splits,
        n_repeat_quaero_corpus=n_repeat_quaero_corpus,
        source_full_lexicon=source_full_lexicon,
        source_lat=source_lat,
        other_full_lexicon=other_full_lexicon,
        other_lat=other_lat,
        other_sabs=other_sabs,
        other_additional_labels=other_additional_labels,
        other_mirror_existing_labels=other_mirror_existing_labels,
        sty_groups=sty_groups,
        filter_before_preprocess=filter_before_preprocess,
        subs=subs,
        max_length=max_length,
        apply_unidecode=apply_unidecode,
        prepend_labels=prepend_labels,
        mentions=mentions,
        return_raw_mentions=return_raw_mentions,
        vocabularies=vocabularies,
    )
    if return_raw_mentions:
        return res
    else:
        train_batcher, vocabularies, train_mentions, train_mention_ids, group_label_mask = res
    quaero_batcher, vocabularies, quaero_mentions, quaero_mention_ids = preprocess_quaero(
        bert_name=bert_name,
        subs=subs,
        apply_unidecode=apply_unidecode,
        max_length=max_length,
        vocabularies=vocabularies,
    )
    return train_batcher, vocabularies, train_mentions, train_mention_ids, group_label_mask, quaero_batcher, quaero_mentions, quaero_mention_ids
