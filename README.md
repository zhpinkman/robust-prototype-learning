This repository contains the code for the paper "Robust Text Classification: Analyzing Prototype-Based Networks" accepted to EMNLP 2024. In the following, different parts of the code and repository are explained for reproducibility purposes. As some modifications were made to the TextAttack package for the attacks to be executed on the prototype-based networks, the modified version of this package can be accessed from the following link: [TextAttack](https://github.com/zhpinkman/custom-textattack).

## Requirements

Please install the requirements by running the following command:

```
conda env create -f environment.yml
```

## Repository Structure

The structure of the repository can be broken down to three parts: (1) adversarial attacks and dataset creation, (2) training and evaluation of the Prototype-based networks with transformer backbone, and (3) training and evaluation of the baseline models and vanilla models with Transformer backbones.

`datasets` directory contains the datasets used in the paper. This includes SST-2 from AdvGLUE, and IMDB, AGNews, and DBPedia that were further attacked by the authors. The script to run the attacks and extract the adversarial perturbed examples is also included in this directory.

`normal_models` directory contains the code for training and evaluating the baseline models and vanilla models with Transformer backbones. 

`PBN_Transformer` directory contains the code for training and evaluating the Prototype-based networks with Transformer backbone.

`gpt_experiments` directory contains the code for using GPT3.5 as a baseline model.

## Visualizing the Results

All the visualizations can be accessed for the static and targeted attacks separately in the two notebooks `analysis_on_dynamic_attacks.ipynb` and `analysis_on_static_attacks.ipynb` respectively.