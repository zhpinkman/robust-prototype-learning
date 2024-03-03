for dataset in "dbpedia"; do # "imdb" "ag_news"
    # dataset="olid"
    for attack_type in "bae" "a2t"; do # the ones that are already done "textfooler" "textbugger" "deepwordbug" "pwws"
        # the ones that take really long "checklist" "hotflip" "iga" "input_reduction" "kuleshov" "swarm" "clare" "pruthi"
        if [ "$dataset" = "ag_news" ]; then
            for model_checkpoint in "../normal_models/models/ag_news_prajjwal1/bert-medium" "textattack/roberta-base-ag-news" "textattack/bert-base-uncased-ag-news" "andi611/distilbert-base-uncased-ner-agnews" "../normal_models/models/ag_news_ModelTC/bart-base-mnli" "../normal_models/models/ag_news_google/electra-base-discriminator"; do
                echo " Attack type: " $attack_type
                echo " Dataset: " $dataset
                echo " Model checkpoint: " $model_checkpoint
                CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack.py \
                    --dataset $dataset \
                    --attack_type $attack_type \
                    --model_checkpoint $model_checkpoint \
                    --mode "attack"
            done
        elif [ "$dataset" = "imdb" ]; then
            for model_checkpoint in "../normal_models/models/imdb_prajjwal1/bert-medium" "textattack/bert-base-uncased-imdb" "textattack/distilbert-base-uncased-imdb" "textattack/albert-base-v2-imdb" "textattack/roberta-base-imdb" "../normal_models/models/imdb_ModelTC/bart-base-mnli" "../normal_models/models/imdb_google/electra-base-discriminator"; do
                echo " Attack type: " $attack_type
                echo " Dataset: " $dataset
                echo " Model checkpoint: " $model_checkpoint
                CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack.py \
                    --dataset $dataset \
                    --attack_type $attack_type \
                    --model_checkpoint $model_checkpoint \
                    --mode "attack"
            done
        elif [ "$dataset" = "dbpedia" ]; then
            for model_checkpoint in "../normal_models/models/dbpedia_prajjwal1/bert-medium" "../normal_models/models/dbpedia_bert-base-uncased" "../normal_models/models/dbpedia_distilbert-base-uncased" "../normal_models/models/dbpedia_roberta-base" "../normal_models/models/dbpedia_ModelTC/bart-base-mnli" "../normal_models/models/dbpedia_google/electra-base-discriminator"; do
                echo " Attack type: " $attack_type
                echo " Dataset: " $dataset
                echo " Model checkpoint: " $model_checkpoint
                CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack.py \
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
done

# elif [ "$dataset" = "olid" ]; then
#     # for model_checkpoint in "../normal_models/models/dbpedia_ModelTC/bart-base-mnli"; do
#     for model_checkpoint in "../normal_models/models/olid_bert-base-uncased" "../normal_models/models/olid_distilbert-base-uncased" "../normal_models/models/olid_roberta-base"; do
#         echo " Attack type: " $attack_type
#         echo " Dataset: " $dataset
#         echo " Model checkpoint: " $model_checkpoint
#         CUDA_VISIBLE_DEVICES=2,3,4,5,7 python adv_attack.py \
#             --dataset $dataset \
#             --attack_type $attack_type \
#             --model_checkpoint $model_checkpoint \
#             --mode "attack"
#     done
