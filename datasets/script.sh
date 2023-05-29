for attack_type in "hotflip" "iga" "pso" "kuleshov" "alzantot" "textfooler" "checklist" "pwws" "textbugger"; do
    echo "Running attack: $attack_type"
    CUDA_VISIBLE_DEVICES=4 python adv_attack.py --attack_type "$attack_type"
done
