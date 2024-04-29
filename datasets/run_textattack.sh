# attack_type=$1
# subset_models=$2
for dataset in "dbpedia" "imdb" "ag_news"; do #
    for subset_models in "1" "2" "3"; do
        # dataset="olid"
        for attack_type in "bae" "textfooler" "textbugger" "deepwordbug" "pwws"; do # the ones that are already done
            # the ones that take really long "checklist" "hotflip" "iga" "input_reduction" "kuleshov" "swarm" "clare" "pruthi" "a2t"
            if [ "$dataset" = "ag_news" ]; then
                if [ "$subset_models" = "1" ]; then
                    echo "Running subset 1"
                    for model_checkpoint in "../normal_models/models/ag_news_prajjwal1/bert-medium" "textattack/roberta-base-ag-news"; do
                        echo " Attack type: " $attack_type
                        echo " Dataset: " $dataset
                        echo " Model checkpoint: " $model_checkpoint
                        CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack.py \
                            --dataset $dataset \
                            --attack_type $attack_type \
                            --model_checkpoint $model_checkpoint \
                            --mode "read"
                    done

                elif [ "$subset_models" = "2" ]; then
                    echo "Running subset 2"
                    for model_checkpoint in "textattack/bert-base-uncased-ag-news" "andi611/distilbert-base-uncased-ner-agnews"; do
                        echo " Attack type: " $attack_type
                        echo " Dataset: " $dataset
                        echo " Model checkpoint: " $model_checkpoint
                        CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack.py \
                            --dataset $dataset \
                            --attack_type $attack_type \
                            --model_checkpoint $model_checkpoint \
                            --mode "read"
                    done

                elif [ "$subset_models" = "3" ]; then
                    echo "Running subset 3"
                    for model_checkpoint in "../normal_models/models/ag_news_ModelTC/bart-base-mnli" "../normal_models/models/ag_news_google/electra-base-discriminator"; do
                        echo " Attack type: " $attack_type
                        echo " Dataset: " $dataset
                        echo " Model checkpoint: " $model_checkpoint
                        CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack.py \
                            --dataset $dataset \
                            --attack_type $attack_type \
                            --model_checkpoint $model_checkpoint \
                            --mode "read"
                    done
                fi

            elif [ "$dataset" = "imdb" ]; then
                if [ "$subset_models" = "1" ]; then
                    for model_checkpoint in "../normal_models/models/imdb_prajjwal1/bert-medium" "textattack/bert-base-uncased-imdb" "textattack/distilbert-base-uncased-imdb" "textattack/albert-base-v2-imdb" "textattack/roberta-base-imdb" "../normal_models/models/imdb_ModelTC/bart-base-mnli" "../normal_models/models/imdb_google/electra-base-discriminator"; do
                        echo " Attack type: " $attack_type
                        echo " Dataset: " $dataset
                        echo " Model checkpoint: " $model_checkpoint
                        CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack.py \
                            --dataset $dataset \
                            --attack_type $attack_type \
                            --model_checkpoint $model_checkpoint \
                            --mode "read"
                    done
                fi
            elif [ "$dataset" = "dbpedia" ]; then
                if [ "$subset_models" = "1" ]; then
                    echo "Running subset 1"
                    for model_checkpoint in "../normal_models/models/dbpedia_prajjwal1/bert-medium" "../normal_models/models/dbpedia_bert-base-uncased"; do
                        echo " Attack type: " $attack_type
                        echo " Dataset: " $dataset
                        echo " Model checkpoint: " $model_checkpoint
                        CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack.py \
                            --dataset $dataset \
                            --attack_type $attack_type \
                            --model_checkpoint $model_checkpoint \
                            --mode "read"
                    done
                elif [ "$subset_models" = "2" ]; then
                    echo "Running subset 2"
                    for model_checkpoint in "../normal_models/models/dbpedia_distilbert-base-uncased" "../normal_models/models/dbpedia_roberta-base"; do
                        echo " Attack type: " $attack_type
                        echo " Dataset: " $dataset
                        echo " Model checkpoint: " $model_checkpoint
                        CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack.py \
                            --dataset $dataset \
                            --attack_type $attack_type \
                            --model_checkpoint $model_checkpoint \
                            --mode "read"
                    done
                elif [ "$subset_models" = "3" ]; then
                    echo "Running subset 3"
                    for model_checkpoint in "../normal_models/models/dbpedia_ModelTC/bart-base-mnli" "../normal_models/models/dbpedia_google/electra-base-discriminator"; do
                        echo " Attack type: " $attack_type
                        echo " Dataset: " $dataset
                        echo " Model checkpoint: " $model_checkpoint
                        CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack.py \
                            --dataset $dataset \
                            --attack_type $attack_type \
                            --model_checkpoint $model_checkpoint \
                            --mode "read"
                    done
                fi
            else
                echo "Invalid dataset"
                exit 1
            fi
        done
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
#             --mode "read"
#     done
