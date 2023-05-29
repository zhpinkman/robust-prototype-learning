for dataset in "ag_news" "imdb"; do
    for attack_type in "textfooler" "textbugger"; do
        python adv_attack.py --dataset $dataset --attack_type $attack_type
    done
done
