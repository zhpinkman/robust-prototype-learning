for architecture in "BART" "BERT" "ELECTRA"; do
    for dataset in "imdb" "dbpedia" "ag_news"; do
        if [ "$dataset" = "imdb" ]; then
            batch_size=512
        elif [ "$dataset" = "dbpedia" ]; then
            batch_size=512
        else
            batch_size=64
        fi

        for attack_type in "bae" "textfooler" "textbugger" "deepwordbug" "pwws"; do # the ones that are already done
            # for model_checkpoint in "textattack/bert-base-uncased-imdb" "textattack/albert-base-v2-imdb" "textattack/roberta-base-imdb"; do
            echo " Attack type: " $attack_type
            echo " Dataset: " $dataset
            echo " Batch size" $batch_size
            for p1_lamb in 0.9; do
                for p2_lamb in 0.9; do
                    for p3_lamb in 0.9; do
                        for num_proto in 2 4 8 16 64; do
                            TEXTATTACK_MAX_LENGTH=${batch_size} CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack_pbn.py \
                                --dataset $dataset \
                                --attack_type $attack_type \
                                --model_checkpoint "${architecture}_${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}_${num_proto}" \
                                --mode "attack" \
                                --batch_size 128 \
                                --architecture $architecture \
                                --num_prototypes $num_proto
                        done
                    done
                done
            done

            for p1_lamb in 0.9; do
                for p2_lamb in 0.9; do
                    for p3_lamb in 0.0 0.9 10.0; do
                        for num_proto in 16; do
                            TEXTATTACK_MAX_LENGTH=${batch_size} CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack_pbn.py \
                                --dataset $dataset \
                                --attack_type $attack_type \
                                --model_checkpoint "${architecture}_${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}_${num_proto}" \
                                --mode "attack" \
                                --batch_size 128 \
                                --architecture $architecture \
                                --num_prototypes $num_proto
                        done
                    done
                done
            done
            for p1_lamb in 0.9; do
                for p2_lamb in 0.0 0.9 10.0; do
                    for p3_lamb in 0.9; do
                        for num_proto in 16; do
                            TEXTATTACK_MAX_LENGTH=${batch_size} CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack_pbn.py \
                                --dataset $dataset \
                                --attack_type $attack_type \
                                --model_checkpoint "${architecture}_${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}_${num_proto}" \
                                --mode "attack" \
                                --batch_size 128 \
                                --architecture $architecture \
                                --num_prototypes $num_proto
                        done
                    done
                done
            done
            for p1_lamb in 0.0 0.9 10.0; do
                for p2_lamb in 0.9; do
                    for p3_lamb in 0.9; do
                        for num_proto in 16; do
                            TEXTATTACK_MAX_LENGTH=${batch_size} CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python adv_attack_pbn.py \
                                --dataset $dataset \
                                --attack_type $attack_type \
                                --model_checkpoint "${architecture}_${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}_${num_proto}" \
                                --mode "attack" \
                                --batch_size 128 \
                                --architecture $architecture \
                                --num_prototypes $num_proto
                        done
                    done
                done
            done
        done
    done
done
