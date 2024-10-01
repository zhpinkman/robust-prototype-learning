This directory contains the datasets used in the paper. This includes SST-2 from AdvGLUE, and IMDB, AGNews, and DBPedia that were further attacked by the authors. The script to run the attacks and extract the adversarial perturbed examples is also included in this directory.


## Structure of Data

Each dataset has its own directory named "${dataset_name}_dataset" with `dataset_name` being "imdb", "ag_news", "dbpedia", and "sst2". Inside each directory, different splits of the datasets are presented, categorized with the beginning of their names and ending with the attack type that is associated with them. For instance, `adv_paraphrased.csv` corresponds to the split of the data that is perturbed using paraphrasing strategy, and `test_paraphrased.csv` corresponds to the same split of the data that contains the unperturbed examples presented in the `adv_paraphrased.csv`. In each directory, `train.csv`, `val.csv`, and `test.csv` correspond to the original splits of the data that were fetched from `Huggingface.com`.

## Scripts for Running the Char-level and Word-level Attacks

`adv_attack.py` contains the code for running the attacks that are covered by TextAttack. In order to run the attacks, you need to specify the type of the attack that has to be applied on the dataset, the dataset, and the model checkpoint that attack will be targeting. Examples of all values that can be passed to this file are presented in the `run_textattack.sh` file that we prepared to run the attacks on datasets and different checkpoints more systematically.

A sample script for attacking the model with checkpoint `textattack/roberta-base-ag-news`, on `ag_news` dataset with "textbugger" strategy is presented below, which applies the attacks on the mentioned dataset and stores the logs and pertubed examples in `logs/` and `summaries/` directory, respectively.

```

CUDA_VISIBLE_DEVICES=4,5,6 python adv_attack.py \
    --dataset "ag_news" \
    --attack_type "textbugger" \
    --model_checkpoint "textattack/roberta-base-ag-news" \
    --mode "attack"

```


As it is mentioned in the paper, we use target three different models for each attack to ensure the transferability of the perturbations. After getting the perturbed examples for each attack and checkpoint, you can use `aggregate_attacks.py` without any arguments needed to pass to it.


## Scripts for Running the Sentence-level Attacks

For applying the sentence-level attacks, we chose paraphrasing attacks. We find the perturbations by prompting a very large language model (GPT 3.5). The code for paraphrasing is provided in `paraphrase.py` and the test values to be passed to this file are provided in `run_paraphrasing.sh`.