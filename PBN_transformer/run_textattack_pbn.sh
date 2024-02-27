p1_lamb=0.9
p2_lamb=0.9
p3_lamb=0.9
architecture="BERT"
# dataset="ag_news"
# attack_type="textfooler"

for dataset in "ag_news"; do # "imdb" "dbpedia"; do
    for attack_type in "textbugger" "textfooler"; do
        if [ "$dataset" = "ag_news" ]; then
            # for model_checkpoint in "textattack/roberta-base-ag-news" "textattack/bert-base-uncased-ag-news" "andi611/distilbert-base-uncased-ner-agnews"; do
            echo " Attack type: " $attack_type
            echo " Dataset: " $dataset

            TEXTATTACK_MAX_LENGTH=64 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack_pbn.py \
                --dataset $dataset \
                --attack_type $attack_type \
                --model_checkpoint "${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}_BERT_M" \
                --mode "attack" \
                --batch_size 512 \
                --architecture $architecture
            # done
        elif [ "$dataset" = "imdb" ]; then
            # for model_checkpoint in "textattack/bert-base-uncased-imdb" "textattack/albert-base-v2-imdb" "textattack/roberta-base-imdb"; do
            echo " Attack type: " $attack_type
            echo " Dataset: " $dataset

            TEXTATTACK_MAX_LENGTH=512 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack_pbn.py \
                --dataset $dataset \
                --attack_type $attack_type \
                --model_checkpoint "${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}_BERT_M" \
                --mode "attack" \
                --batch_size 512 \
                --architecture $architecture
            # done
        elif [ "$dataset" = "dbpedia" ]; then
            # for model_checkpoint in "../normal_models/models/dbpedia_bert-base-uncased" "../normal_models/models/dbpedia_distilbert-base-uncased" "../normal_models/models/dbpedia_roberta-base"; do
            echo " Attack type: " $attack_type
            echo " Dataset: " $dataset

            TEXTATTACK_MAX_LENGTH=512 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack_pbn.py \
                --dataset $dataset \
                --attack_type $attack_type \
                --model_checkpoint "${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}_BERT_M" \
                --mode "attack" \
                --batch_size 512 \
                --architecture $architecture
            # done
        else
            echo "Invalid dataset"
            exit 1
        fi
    done
done
