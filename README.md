Deep Multilingual Normalization
===============================

This repository hosts the code for our [Medical concept normalization in French using multilingual terminologies and contextual embeddings](https://doi.org/10.1016/j.jbi.2021.103684) article. It was recently reimplemented using [edsnlp](https://github.com/aphp/edsnlp).

If this method is useful to you, please consider citing our article, and/or giving a star to this repository :

```bibtex
@article{wajsburt2021medical,
    title = {Medical concept normalization in French using multilingual terminologies and contextual embeddings},
    journal = {Journal of Biomedical Informatics},
    volume = {114},
    pages = {103684},
    year = {2021},
    issn = {1532-0464},
    doi = {https://doi.org/10.1016/j.jbi.2021.103684},
    url = {https://www.sciencedirect.com/science/article/pii/S1532046421000137},
    author = {Perceval Wajsbürt and Arnaud Sarfati and Xavier Tannier},
    keywords = {Natural language processing, Information extraction, Medical concept normalization, Multilingual representation},
}
```

## Install

We recommend you use [`poetry`](https://python-poetry.org/docs/#installing-with-pipx) to install the dependencies from the lock file.

```bash
# Clone the repo
git clone https://github.com/percevalw/mlg_norm.git
cd mlg_norm

# Install the dependencies with poetry (or use pip otherwise)
poetry install
# pip install -e .
```

## Downloading the UMLS

You will need to download the UMLS version to run this method. For instance, to replicate our results on the Quaero corpus, you will need the 2014AB version. Here are the steps to load the UMLS:

1. Download and unzip the `2014ab-1-meta.nlm` file (it's really a zip with a different extension) under the *2014AB UMLS Full Release Files* section at [https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html#2014AB_full](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html#2014AB_full)
2. Enter the `2014AB/META` folder and unzip MRCONSO and MRSTY

    ```
    gunzip MRCONSO.RRF.*.gz MRSTY.RRF.*.gz
    ```
3. Concatenate the multiple MRCONSO files:

    ```
    cat MRCONSO.RRF.aa MRCONSO.RRF.ab > MRCONSO.RRF
    ```
4. Move `MRCONSO.RRF`, `MRSTY.RRF` and `resources/sty_groups.tsv` to the `data/umls/2014AB` folder.

## Downloading Quaero

Download Quaero in BRAT format, unzip it and move the `QUAERO_FrenchMed/corpus` folder to `data/dataset`.

```bash
wget https://quaerofrenchmed.limsi.fr/QUAERO_FrenchMed_brat.zip
unzip QUAERO_FrenchMed_brat.zip
mv QUAERO_FrenchMed/corpus data/dataset
```

## Train and evaluate a model

Our method is composed of two steps:

- Pre-training, to learn multilingual representations and produce similar representation for synonyms of a same concept:

    ```bash
    python scripts/train.py pretrain --config configs/config.cfg
    ```

- Short classifier training. This will probe the pre-trained embedding and finetune the concepts weights.

    ```bash
    python scripts/train.py train_classifier --config configs/config.cfg
    ```

Finally, you can evaluate the model:

```bash
python scripts/evaluate.py evaluate --config configs/config.cfg
```

Consider changing the [`configs/config.cfg`](/configs/config.cfg) to fit your needs.
