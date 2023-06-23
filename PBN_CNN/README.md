This directory contains the code for training and evaluating the Prototype-based networks with CNN backbone.




## Requirements

Please install the requirements by running the following command:

```
conda env create -f conda_environment.yml
```
## Training and Evaluating the Models


Be sure to run the scripts and other files from `src` directory. The `train_custom.py` contains the code for training and `evaluate_dataset.py` contains the code for evaluation. The script `run_training.sh` contains the script for running the training code. For example, to train the version of ProtoCNN in the paper on the imdb dataset, run the following:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python3 train_custom.py \
    --type="softmax" \
    --dataset_name "imdb" \
    --num_labels=2 \
```

Once trained, you can evaluate it with the script in `evaluate_dataset.sh` by modifying the contents to:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python evaluate_dataset.py \
    --dataset_dir="../../datasets/imdb_dataset" \
    --model_checkpoint="/path/to/checkpoint.pt" \
    --name "softmax" \
    --num_labels=2
```

## Pretrained model weights

Models will be saved in the directory `src/models/`. In case you wanted to use the model that we have trained, you can put them in the same directory. 
