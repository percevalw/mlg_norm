Deep Multilingual Normalization
===============================

This repository will host the code used our "Medical concept normalization in French using multilingual terminologies and contextual embeddings" article.

# Install

```
# Install packages
conda create -n deep_multilingual_normalization python==3.6.5
conda activate deep_multilingual_normalization
pip install torch==1.5.1 tensorflow nlstruct==0.0.2 transformers==2.9.0 faiss-gpu

# Clone the repo
git clone https://github.com/percevalw/deep_multilingual_normalization.git
```

# Data

You will need to download the UMLS version 2014AB to replicate our results:
1. Download all the files under the *2014AB UMLS Active Release Files* section at [https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html#2014AB_active](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html#2014AB_active)
2. Unzip `mmsys.zip`
3. Copy all the files downloaded files (`mmsys.zip` included !) inside the new mmsys folder at the top level
4. Run the main command depending on your OS (ex: `run_mac.command`)
5. Select *Install UMLS*
6. Remember the destination path, we will need to move some files from it later
7. Select *New configuration*
8. Go to *Source List* and deselect all lines (ie do not exclude any source, we will filter the UMLS ourselves during the preprocessing)
9. Select *Done > Begin Subset*... and wait
10. Locate the `MRCONSO.RRF` and `MRSTY.RRF` and move them under a new folder `SOME_PATH/resources/umls/2014AB/`, you will need to fill `SOME_PATH/resources` and `SOME_PATH/cache` in a config file when prompted later
11. Copy the `sty_groups.tsv` file to `SOME_PATH/resources/umls/`

# Run

If you want, you can change parameters directly in the scripts.

To run our experiment on Quaero FrenchMed CLEF 15, run the following command:
```bash
python en_only-unsup14-clef15.py
```
