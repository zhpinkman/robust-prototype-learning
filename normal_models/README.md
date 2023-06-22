This directory contains the code for training and evaluating the baseline models and vanilla models with both Transformer and CNN backbones. 



## Requirements

Please install the requirements by running the following command:

```
conda env create -f conda_environment.yml
```

## Training and Evaluating the Models

`vanilla_model.py` python file includes the code for training and evaluating baseline Transformer models that is implemented with HuggingFace Trainer pipeline. Given a dataset directory that has to contain the train, validation, and test splits, and a model checkpoint, this file can train the model and evaluates the model on all splits that can be found in the dataset directory. Sample values that can be passed and we used in our experiments are provided in `train_vanilla_model.sh`.