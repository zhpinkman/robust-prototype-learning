This directory contains the code for training and evaluating the Prototype-based networks with Transformer backbone that is based on the implementation of the paper, "ProtoTEx: Explaining Model Decisions with Prototype Tensors".



## Training and Evaluating the Models

`main.py` and `training.py` python file contains the code for training the model and `evaluate_model.py` contains the code for evaluating a fine-tuned model. The script, `train_eval.sh` contains the code for using the mentioned python files that have sample arguments passed to them as well. 

A sample script for training the prototype-based network with a `Electra` bone, batch size of 64 on `dbpedia` dataset with the given data directory and a directory path to save the model checkpoints i presented below:

```
CUDA_VISIBLE_DEVICES=1,2,3 python main.py \
    --batch_size 64 \
    --dataset "dbpedia" \
    --data_dir "datasets/dbpedia_dataset" \
    --p1_lamb 0.9 \
    --p2_lamb 0.9 \
    --p3_lamb 0.9 \
    --architecture "ELECTRA" \
    --modelname "dbpedia_model_ELECTRA"
```

To evaluate the trained model, you can use the script below that evaluates the model on all csv files 


```
CUDA_VISIBLE_DEVICES=1,2,3 python evaluate_model.py \
    --batch_size 512 \
    --dataset "dbpedia" \
    --data_dir "datasets/dbpedia_dataset" \
    --modelname "dbpedia_model_ELECTRA" \
    --architecture $architecture \
    --num_prototypes $num_proto
```

## Pretrained model weights

All pretrained model weights are available upon request. Please contact the authors for the pretrained model weights.


## Adversarial Attacks on the Model

The code for the last experiment of the paper in which we adversarially attack PBNs can be found in `adv_attack_pbn.py` file with the sample scripts that are included in `run_textattack_pbn.sh`. An instance of the script is presented below:

```
CUDA_VISIBLE_DEVICES=5,6,7 python adv_attack_pbn.py \
    --dataset "ag_news" \
    --attack_type "textfooler" \
    --model_checkpoint "${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}" \
    --mode "attack" \
    --batch_size 512 \
    --architecture "BART"
```


Furthermore, to evaluate all the models and gathering the log files that track each models' performance as well as the developed prototypes on different datasets and their adversarial perturbations, you can use the files: `evaluate_all_the_pbn_models_prototypes.py` and `evaluate_all_the_pbn_models.py`.


Also, the code to evaluate LLMs on the perturbations can be found in `evaluate_all_perturbations_gpt.py` and `evaluate_all_perturbations_llama.py`.

