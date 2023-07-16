CUDA_VISIBLE_DEVICES=4 python adv_attack_protocnn.py \
    --dataset="imdb" \
    --model_checkpoint="checkpoints/imdb_dataset/checkpoint.ckpt" \
    --name "normal" \
    --attack_type "textbugger" \
    --mode "attack"
