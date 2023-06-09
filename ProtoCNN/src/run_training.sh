pwd

eval "$(conda shell.bash hook)"
conda activate prototype_learning
# nvidia-smi

for type in "without_clustering_loss" "with_linear" "without_dynamic"; do
    echo "----------- $type -----------"
    CUDA_VISIBLE_DEVICES=3,4,5,6,7 python3 train_custom.py \
        --type $type \
        --dataset_name="dbpedia_dataset" \
        --num_labels=9
done

conda deactivate