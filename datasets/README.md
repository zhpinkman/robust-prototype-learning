This directory contains the datasets used in the paper. This includes SST-2 from AdvGLUE, and IMDB, AGNews, and DBPedia that were further attacked by the authors. The script to run the attacks and extract the adversarial perturbed examples is also included in this directory.


## Requirements

Please install the requirements by running the following command:

```
conda env create -f conda_environment.yml
```

## Scripts for Running the Char-level and Word-level Attacks

`adv_attack.py` contains the code for running the attacks that are covered by TextAttack. In order to run the attacks, you need to specify the type of the attack that has to be applied on the dataset, the dataset, and the model checkpoint that attack will be targetting. Examples of all values that can be passed to this file are presented in the `run_textattack.sh` file that we prepared to run the attacks on datasets and different checkpoints more systematically.


As it is mentioned in the paper, we use target three different models for each attack to ensure the transferability of the perturbations. After getting the perturbed examples for each attack and checkpoint, you can use `aggregate_attacks.py`


## Scripts for Running the Sentence-level Attacks

For applying the sentence-level attacks, we chose paraphrasing attacks. We find the perturbations by prompting a very large language model (GPT 3.5). The code for paraphrasing is provided in `paraphrase.py` and the test values to be passed to this file are provided in `run_paraphrasing.sh`.