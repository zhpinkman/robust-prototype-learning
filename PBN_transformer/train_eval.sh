################################ Training ################################

dataset=$2
echo "Mode" $1
architecture="BART"

if [ "$1" = "train" ]; then

    for p1_lamb in 0.9; do
        for p2_lamb in 0.9; do
            for p3_lamb in 0.9; do

                WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=$3 python main.py \
                    --batch_size $4 \
                    --dataset $dataset \
                    --data_dir "../datasets/${dataset}_dataset" \
                    --p1_lamb $p1_lamb \
                    --p2_lamb $p2_lamb \
                    --p3_lamb $p3_lamb \
                    --architecture $architecture \
                    --modelname "${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}_${architecture}"
            done
        done
    done

################################ inference ################################

elif [ "$1" = "inference" ]; then
    p1_lamb=0.0
    p2_lamb=0.9
    p3_lamb=0.9
    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=$3 python inference_and_explanations.py \
        --batch_size 256 \
        --dataset $dataset \
        --data_dir "../datasets/${dataset}_dataset" \
        --modelname "${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}"

################################ Testing ################################

elif [ "$1" = "test" ]; then

    p1_lamb=0.9
    p2_lamb=0.9
    p3_lamb=0.9
    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=$3 python evaluate_model.py \
        --batch_size 512 \
        --dataset $dataset \
        --data_dir "../datasets/${dataset}_dataset" \
        --architecture $architecture \
        --modelname "${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}_${architecture}"

else

    echo "Invalid mode"

fi
