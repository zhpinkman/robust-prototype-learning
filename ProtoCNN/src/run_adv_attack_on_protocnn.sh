CUDA_VISIBLE_DEVICES=6 python adv_attack_protocnn.py \
    --dataset="ag_news" \
    --model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/ag_news/ag_news_epoch_0=00-val_loss_0=0.6478-val_acc_0=0.8260.ckpt" \
    --name "normal" \
    --num_labels=4 \
    --attack_type "deepwordbug" \
    --mode "attack"
