# Latin WSD experiments

This repository contains the code and the preliminary results of word sense disambiguation experiments in Latin.

The repository itself is a pseudo-fork of the original [LatinBERT repository](https://github.com/dbamman/latin-bert) and it closely follows the word sense disambiguation setting of [Bamman and Burns (2020)](https://arxiv.org/abs/2009.10053), while using their BERT model pre-trained on latin corpora as described in the original paper.

Compared with Bamman and Burns (2020), we used all macro-senses from the Lewis and Short Latin-English dictionary. We ran the algorithm on the LatinISE corpus (McGillivray and Kilgarriff 2013).

B. McGillivray, A. Kilgarriff, Tools for historical corpus research, and a corpus of Latin, in: P. Bennett, M. Durrell, S. Scheible, R. J. Whitt (Eds.), New Methods in Historical Corpus Linguistics, Narr, Tübingen, 2013.

### Install

*Tested on Python 3.8.12 and 3.7.12.*

1.) Create a [conda environment](https://www.anaconda.com/download/) (optional):

```sh
conda create --name latinbert python=3
conda activate latinbert
```

2.) Install PyTorch according to your own system requirements (GPU vs. CPU, CUDA version): [https://pytorch.org](https://pytorch.org).


3.) Install the remaining libraries:


```sh
pip install -r requirements.txt
```

4.) Install Latin tokenizer models:

```sh
python3 -c "from cltk.data.fetch import FetchCorpus; corpus_downloader = FetchCorpus(language='lat');corpus_downloader.import_corpus('lat_models_cltk')"
```

### Use

All the code related to the word sense disambiguation analysis is contained in the wsd folder of this same repository, while this parent directory contains the pre-trained LatinBERT and other utils functions. See https://github.com/Ighina/Latin-ISE-WSD/tree/master/wsd#readme for details.
