This repository contains the code for the paper "Robust Text Classification: Analyzing Prototype-Based Networks". In the following, different parts of the code and repository are explained for reproducibility purposes.

## Repository Structure

The structure of the repository can be broken down to four parts: (1) adversarial attacks and dataset creation, (2) training and evaluation of the Prototype-based networks with transformer backbone, (3) training and evaluation of the Prototype-based networks with CNN backbone, and (4) training and evaluation of the baseline models and vanilla models with Transformer backbones.

`datasets` directory contains the datasets used in the paper. This includes SST-2 from AdvGLUE, and IMDB, AGNews, and DBPedia that were further attacked by the authors. The script to run the attacks and extract the adversarial perturbed examples is also included in this directory.

`normal_models` directory contains the code for training and evaluating the baseline models and vanilla models with Transformer backbones. 

`PBN_CNN` directory contains the code for training and evaluating the Prototype-based networks with CNN backbone.

`PBN_Transformer` directory contains the code for training and evaluating the Prototype-based networks with Transformer backbone.

`gpt_experiments` directory contains the code for using GPT3.5 as a baseline model.

## Requirements

Since the purpose of each subdirectory as well as the tools and packages used in each of them are different, the requirements are also different and provided separately in each subdirectory as a conda environment file called `conda_environment.yml` that you can use to create a conda environment with the required packages by running the following command:

```
conda env create -f conda_environment.yml
```

## Visualizing the Results

All the results by all the experiments are saved in the `Results - Sheet1.csv` file. In this csv file, `Model` column shows the model that is being tested, `comments` note on the variation of the model, `dataset` shows the dataset of interest, `dataset type` specifies the perturbation strategy used for perturbation, and `Precision Performance`, `Recall Performance`, `F1 performance` columns show the performance of the model on the dataset. You can use the `result_analyzer.ipynb` notebook to explore and visualize the results the same way we did in the paper.
