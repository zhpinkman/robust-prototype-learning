pwd

eval "$(conda shell.bash hook)"
conda activate prototype_learning
# nvidia-smi

# for type in "with_linear" "without_clustering_loss" "without_separation_loss" "without_dynamic"; do
# for type in "bigger_softmax"; do
for dataset in "dbpedia_dataset" "sst2_dataset" "ag_news_dataset"; do
    echo "----------- $dataset -----------"
    CUDA_VISIBLE_DEVICES=1,4,5 python3 train_custom.py \
        --type="bigger" \
        --dataset_name $dataset \
        --num_labels=2 \
        --use_bigger
done

conda deactivate


CUDA_VISIBLE_DEVICES=1,4,5 python3 train_custom.py --type="bigger" --dataset_name="dbpedia_dataset" --num_labels=9 --use_bigger