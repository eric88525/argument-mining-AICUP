#! /bin/bash
BASEDIR=$(pwd)

python "${BASEDIR}/multi_model_vote.py" \
    --vote_output "${BASEDIR}/vote.csv" \
    --batch_size "8" \
    --seed "2022" \
    --test_csv_path "${BASEDIR}/dataset/batch_answer.csv" \
    --max_seq_len "1024"
