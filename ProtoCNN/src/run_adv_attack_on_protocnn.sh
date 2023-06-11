CUDA_VISIBLE_DEVICES=4 python adv_attack_protocnn.py \
    --dataset="imdb" \
    --model_checkpoint="/scratch/darshan/prototype-learning/robust-prototype-learning/ProtoCNN/src/checkpoints/imdb_dataset/softmax-epoch_0=00-val_loss_0=0.4311-val_acc_0=0.8279.ckpt" \
    --name "normal" \
    --attack_type "textbugger" \
    --mode "attack"
