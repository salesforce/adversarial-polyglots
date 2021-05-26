# Adversarial Polyglots
This repository contains code for the paper "[Code-Mixing on Sesame Street: Dawn of the Adversarial Polyglots](https://arxiv.org/abs/2103.09593)" (NAACL-HLT 2021).

Authors: [Samson Tan](https://samsontmr.github.io) and [Shafiq Joty](https://raihanjoty.github.io)


# Usage

## Adversarial Polyglots
Scripts for running `PolyGloss` and `Bumblebee` on NLI and QA datasets (MNLI/SQuAD formats) are in the `attacks` folder. Preprocessing scripts can be found in `scripts`. `PolyGloss` and `Bumblebee` return the _*adversarial*_ examples with the highest and lowest losses. The one that induced the lower loss _(minimally perturbed)_ is usually less perturbed, but the one that induced a higher loss _(maximally perturbed)_ should transfer more successfully to other models.

`PolyGloss` requires a dictionary constructed from the bilingual [MUSE dictionaries](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries). After downloading the dictionaries into a folder under `scripts` labeled `dictionaries`, run `scripts/create_polygloss_dictionaries_from_muse.py`.

`Bumblebee` requires a dictionary/JSON consisting of sentence-translations pairs. The `extract-xnli-sentences-to-dict.py` and `extract-xquad-questions-to-dict.py` scripts in `scripts` can be used to create these dictionaries for files in the MNLI and SQuAD formats (e.g., XNLI and XQuAD). JSONs for the XNLI test set can be found [here](https://github.com/salesforce/adversarial-polyglots-data).

## Code-mixed Adversarial Training
Code for generating code-mixed adversarial training (CAT) examples are in `adversarial-training`. Since the alignment step is the most time-consuming, we decouple it from the example perturbation step. Users can generate only the alignments by using the `--extract_phrases` option or load precomputed alignments via the `phrase_alignments` option.

Similar to `Bumblebee`, `Code-Mixer` requires a dictionary/JSON consisting of sentence-translations pairs. The `extract-xnli-sentences-to-dict.py` and `extract-xquad-questions-to-dict.py` scripts in `scripts` can be used to create these dictionaries for files in the MNLI and SQuAD formats (e.g., XNLI and XQuAD).

# Translated XNLI Data
We translated the [XNLI data](https://cims.nyu.edu/~sbowman/xnli) to 18 other languages using machine-translation systems (see paper for details). Translation script is in `scripts`. Translated data can be found [here](https://github.com/salesforce/adversarial-polyglots-data).

# Citation
Please cite the following if you use the code/data in this repository:
```
@inproceedings{tan-joty-2021-code-mixing,
    title = "Code-Mixing on Sesame Street: {D}awn of the Adversarial Polyglots",
    author = "Tan, Samson and Joty, Shafiq",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.282",
    pages = "3596--3616",
}

```
