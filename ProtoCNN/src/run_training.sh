pwd

eval "$(conda shell.bash hook)"
conda activate prototype_learning
# nvidia-smi

for type in "with_linear" "without_clustering_loss" "without_separation_loss" "without_dynamic"; do
    echo "----------- $type -----------"
    CUDA_VISIBLE_DEVICES=1,3,6,7 python3 train_custom.py \
        --type $type \
        --dataset_name="sst2_dataset" \
        --num_labels=2
done

conda deactivate
