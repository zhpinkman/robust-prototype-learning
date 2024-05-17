################################ Training ################################

echo "Mode" $1
dataset=$2
architecture=$4

# for imdb and dbpedia, the batch size is 32, and for ag_news is 64
if [ "$dataset" == "imdb" ]; then
    batch_size=32
elif [ "$dataset" == "dbpedia" ]; then
    batch_size=32
else
    batch_size=256
fi

if [ "$1" = "train" ]; then

    for p1_lamb in 0.9; do
        for p2_lamb in 0.9; do
            for p3_lamb in 0.9; do
                for num_proto in 2 4 8 16 64; do

                    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=$3 python main.py \
                        --batch_size $batch_size \
                        --dataset $dataset \
                        --data_dir "../datasets/${dataset}_dataset" \
                        --p1_lamb $p1_lamb \
                        --p2_lamb $p2_lamb \
                        --p3_lamb $p3_lamb \
                        --architecture $architecture \
                        --modelname "${architecture}_${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}_${num_proto}" \
                        --num_prototypes $num_proto
                done
            done
        done
    done

    for p1_lamb in 0.9; do
        for p2_lamb in 0.9; do
            for p3_lamb in 0.0 0.9 10.0; do
                for num_proto in 16; do

                    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=$3 python main.py \
                        --batch_size $batch_size \
                        --dataset $dataset \
                        --data_dir "../datasets/${dataset}_dataset" \
                        --p1_lamb $p1_lamb \
                        --p2_lamb $p2_lamb \
                        --p3_lamb $p3_lamb \
                        --architecture $architecture \
                        --modelname "${architecture}_${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}_${num_proto}" \
                        --num_prototypes $num_proto
                done
            done
        done
    done

    for p1_lamb in 0.9; do
        for p2_lamb in 0.0 0.9 10.0; do
            for p3_lamb in 0.9; do
                for num_proto in 16; do

                    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=$3 python main.py \
                        --batch_size $batch_size \
                        --dataset $dataset \
                        --data_dir "../datasets/${dataset}_dataset" \
                        --p1_lamb $p1_lamb \
                        --p2_lamb $p2_lamb \
                        --p3_lamb $p3_lamb \
                        --architecture $architecture \
                        --modelname "${architecture}_${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}_${num_proto}" \
                        --num_prototypes $num_proto
                done
            done
        done
    done

    for p1_lamb in 0.0 0.9 10.0; do
        for p2_lamb in 0.9; do
            for p3_lamb in 0.9; do
                for num_proto in 16; do

                    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=$3 python main.py \
                        --batch_size $batch_size \
                        --dataset $dataset \
                        --data_dir "../datasets/${dataset}_dataset" \
                        --p1_lamb $p1_lamb \
                        --p2_lamb $p2_lamb \
                        --p3_lamb $p3_lamb \
                        --architecture $architecture \
                        --modelname "${architecture}_${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}_${num_proto}" \
                        --num_prototypes $num_proto
                done
            done
        done
    done
# I added architecture before the ${dataset} however, most of the saved models probably won't have this and start with the dataset name

################################ inference ################################

elif [ "$1" = "inference" ]; then
    p1_lamb=0.0
    p2_lamb=0.0
    p3_lamb=0.9
    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=$3 python inference_and_explanations.py \
        --batch_size 256 \
        --dataset $dataset \
        --data_dir "../datasets/${dataset}_dataset" \
        --modelname "${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}"

################################ Testing ################################

elif [ "$1" = "test" ]; then

    p1_lamb=0.0
    p2_lamb=0.0
    p3_lamb=0.0
    num_proto=16
    WANDB_MODE="offline" CUDA_VISIBLE_DEVICES=$3 python evaluate_model.py \
        --batch_size 64 \
        --dataset $dataset \
        --data_dir "../datasets/${dataset}_dataset" \
        --architecture $architecture \
        --modelname "${dataset}_model_${p1_lamb}_${p2_lamb}_${p3_lamb}" \
        --num_prototypes $num_proto

else

    echo "Invalid mode"

fi
