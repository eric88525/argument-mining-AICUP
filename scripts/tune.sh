#!/bin/bash
BASEDIR=$(pwd)

# train
num_workers="8"
seed="2022"
max_seq_len="1024"

# model and data
dataset="deberta_v1"
model="deberta_v1"
pretrained_model="microsoft/deberta-v3-base"

# scheduler
lr_scheduler="linear_warmup"

# loss
loss="ce"
weight_decay="1e-5"

# [tune]
tune_num_samples="10"
tune_num_epochs="10"
label_smoothing="0.0"

# dataset
train_use_dataset="no_same_10fold_2022"
val_use_dataset="super_eval_10fold_2022"
extra_file_folder="len_300_th_0.8"
gpus="0, "

for fold in 1
do
    train_csv_path="${BASEDIR}/my_dataset/${train_use_dataset}/train_${fold}.csv"
    val_csv_path="${BASEDIR}/my_dataset/${val_use_dataset}/val_${fold}.csv"

    exp_name="F${fold}"

    extra_train_file="None"

    pred_csv="${BASEDIR}/tune_${exp_name}.csv"

    python tune.py \
        --exp_name $exp_name \
        --num_workers $num_workers \
        --seed $seed \
        --max_seq_len $max_seq_len \
        --dataset $dataset \
        --model $model \
        --pretrained_model $pretrained_model \
        --pretrained_tokenizer $pretrained_model \
        --pretrained_config $pretrained_model \
        --lr_scheduler $lr_scheduler \
        --loss $loss \
        --weight_decay $weight_decay \
        --train_csv_path $train_csv_path \
        --val_csv_path $val_csv_path \
        --tune_num_samples $tune_num_samples \
        --tune_num_epochs $tune_num_epochs \
        --label_smoothing $label_smoothing \
        --gpus $gpus \
        --pred_csv $pred_csv \
        --extra_train_file $extra_train_file
done

