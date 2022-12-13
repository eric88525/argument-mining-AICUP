#! /bin/bash
BASEDIR=$(pwd)

# train
batch_size="8"
accu_batches="4"
num_workers="8"
seed="2022"
lr="1.7e-4"
max_seq_len="1024"
# model and data
dataset="deberta_v1"
model="deberta_v1"
pretrained_model="microsoft/deberta-v3-base"
# scheduler
lr_scheduler="linear_warmup"
# linear warmup
lr_warmup_ratio="0.104"
# loss
loss="ce"
ce_weight="0.699 1.755"
weight_decay="1e-5"
label_smoothing="0.0"
precision="16"
train_use_dataset="no_same_10fold_2022"
val_use_dataset="super_eval_10fold_2022"
gpus="0, "
max_epochs="10"

for extra_file_folder in "None"  "len_300_th_0.8" 
do
    for fold in 1 2 3 4 5 6 7 8 9 10
    do 

        if [ "$extra_file_folder" = "None" ]; then
            extra_train_file="None"
        else
            extra_train_file="${BASEDIR}/my_dataset/${extra_file_folder}/extra_${fold}.csv"
        fi

        tag="deberta_${extra_file_folder}_${train_use_dataset}_${val_use_dataset}"
        train_csv_path="${BASEDIR}/my_dataset/${train_use_dataset}/train_$fold.csv"
        val_csv_path="${BASEDIR}/my_dataset/${val_use_dataset}/val_$fold.csv"
        log_dir="${BASEDIR}/lightning_logs/${tag}_fold_${fold}"
        exp_name="Fold_${fold}"

        python main.py \
            --exp_name $exp_name \
            --tag $tag \
            --batch_size $batch_size \
            --num_workers $num_workers \
            --seed $seed \
            --lr $lr \
            --max_seq_len $max_seq_len \
            --dataset $dataset \
            --model $model \
            --pretrained_model $pretrained_model \
            --pretrained_tokenizer $pretrained_model \
            --pretrained_config $pretrained_model \
            --lr_scheduler $lr_scheduler \
            --lr_warmup_ratio $lr_warmup_ratio \
            --loss $loss \
            --weight_decay $weight_decay \
            --log_dir $log_dir \
            --train_csv_path $train_csv_path \
            --val_csv_path $val_csv_path \
            --max_epochs $max_epochs \
            --ce_weight $ce_weight \
            --label_smoothing $label_smoothing \
            --accu_batches $accu_batches  \
            --precision $precision \
            --gpus $gpus \
            --extra_train_file $extra_train_file
    done
done
