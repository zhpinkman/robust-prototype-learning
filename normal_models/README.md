This directory contains the code for training and evaluating the baseline models and vanilla models with Transformer backbones. 

## Requirements

Please install the requirements by running the following command:

```
conda env create -f conda_environment.yml
```

## Training and Evaluating the Models

`vanilla_model.py` python file includes the code for training and evaluating baseline Transformer models that is implemented with HuggingFace Trainer pipeline. Given a dataset directory that has to contain the train, validation, and test splits, and a model checkpoint, this file can train the model and evaluates the model on all splits that can be found in the dataset directory. Sample values that can be passed and we used in our experiments are provided in `train_vanilla_model.sh`.

For instance, in the following script, we are training a model with checkpoint `ModelTC/bart-base-mnli`, on `dbpedia` dataset, with the batch size of `64`, logging steps of `100`, and number of epochs of `3`. The model will be saved in `models/dbpedia_ModelTC/bart-base-mnli` directory.


```
CUDA_VISIBLE_DEVICES=1,2,3 python vanilla_model.py \
    --mode train \
    --batch_size 64 \
    --logging_steps 100 \
    --num_epochs 3 \
    --dataset "dbpedia" \
    --data_dir "../datasets/dbpedia_dataset" \
    --model_dir "models/dbpedia_ModelTC/bart-base-mnli \
    --model_checkpoint "ModelTC/bart-base-mnli"

```

For evaluating the model that has been trained above, you can use the following script:

```

CUDA_VISIBLE_DEVICES=1,2,3 python vanilla_model.py \
    --mode test \
    --batch_size 64 \
    --dataset "dbpedia" \
    --data_dir "../datasets/dbpedia_dataset" \
    --model_dir "models/dbpedia_ModelTC/bart-base-mnli

```