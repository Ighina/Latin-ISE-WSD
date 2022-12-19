# Word Sense Disambiguation

This repository contains the code to generate the images contained in figures and figures_prose_poetry, which in turn represent the frequency of senses for each lemma analysed in Latin-ISE across centuries either in general (figures) or divided by macro-genre, i.e. poetry and prose (figures_prose_poetry).

### Training

We fine-tune LatinBERT for each lemma separately in the same way as [Bamman et al. (2020)](https://arxiv.org/abs/2009.10053), but with the difference that we include every macro-sense listed in the relative entry from Lewis & Short dictionary, as opposed to just the first two as in the original paper. Furthermore, we filter out 66 lemmas from the original list of Bamman et al., as we deemed them not interesting from the point of view of diachronic semantic change.

To perform training and save each fine-tuned model, run the code in the run.ipynb notebook in this repository.

### Inference

Once having trained the model, the following cell in the run.ipynb use them to perform inference on Latin-ISE and store them in a file called test_results.csv

We performed additional postprocessing to add metadata to each sentence annotated with the respective sense and we stored the final results in a file named predictions_latinise_with_senses.csv 

### Analysis

To generate our images run the analyse.ipynb notebook in this repository.

