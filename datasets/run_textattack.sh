# for dataset in "imdb" "ag_news" "dbpedia"; do
dataset="dbpedia"
for attack_type in "textfooler"; do
    if [ "$dataset" = "ag_news" ]; then
        for model_checkpoint in "../normal_models/models/ag_news_ModelTC/bart-base-mnli"; do
            # "textattack/roberta-base-ag-news" "textattack/bert-base-uncased-ag-news" "andi611/distilbert-base-uncased-ner-agnews"; do
            echo " Attack type: " $attack_type
            echo " Dataset: " $dataset
            echo " Model checkpoint: " $model_checkpoint
            CUDA_VISIBLE_DEVICES=2,3,4 python adv_attack.py \
                --dataset $dataset \
                --attack_type $attack_type \
                --model_checkpoint $model_checkpoint \
                --mode "attack"
        done
    elif [ "$dataset" = "imdb" ]; then
        for model_checkpoint in "../normal_models/models/imdb_ModelTC/bart-base-mnli"; do
            # "textattack/bert-base-uncased-imdb" "textattack/distilbert-base-uncased-imdb" "textattack/albert-base-v2-imdb" "textattack/roberta-base-imdb"; do
            echo " Attack type: " $attack_type
            echo " Dataset: " $dataset
            echo " Model checkpoint: " $model_checkpoint
            CUDA_VISIBLE_DEVICES=2,3,4 python adv_attack.py \
                --dataset $dataset \
                --attack_type $attack_type \
                --model_checkpoint $model_checkpoint \
                --mode "attack"
        done
    elif [ "$dataset" = "dbpedia" ]; then
        for model_checkpoint in "../normal_models/models/dbpedia_ModelTC/bart-base-mnli"; do
            # "../normal_models/models/dbpedia_bert-base-uncased" "../normal_models/models/dbpedia_distilbert-base-uncased" "../normal_models/models/dbpedia_roberta-base"; do
            echo " Attack type: " $attack_type
            echo " Dataset: " $dataset
            echo " Model checkpoint: " $model_checkpoint
            CUDA_VISIBLE_DEVICES=2,3,4 python adv_attack.py \
                --dataset $dataset \
                --attack_type $attack_type \
                --model_checkpoint $model_checkpoint \
                --mode "attack"
        done
    else
        echo "Invalid dataset"
        exit 1
    fi
done
# done
