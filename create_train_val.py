import random
import numpy as np
import pandas as pd
import os


def create_train_val(official_csv, save_to="dataset", seed=None):

    try:
        df = pd.read_csv(official_csv)
    except:
        raise FileNotFoundError("official csv file not found")

    df = df.drop_duplicates(subset=["s", "q", "r", "qq", "rr"], keep='first')
    unique_ids = df.id.unique()

    if seed != None:
        print(f"Shuffle dataset, seed : {seed}")
        np.random.seed(seed)
        random.seed(seed)
        np.random.shuffle(unique_ids)

    K = 10
    eval_data_count = len(unique_ids) // K

    print(f"unique id : {len(unique_ids)}")

    train_folder = os.path.join(save_to, "train")
    val_folder = os.path.join(save_to, "val")

    if not os.path.isdir(train_folder):
        os.makedirs(train_folder)

    if not os.path.isdir(val_folder):
        os.makedirs(val_folder)

    for i in range(K):

        val_ids = unique_ids[i*eval_data_count:i *
                             eval_data_count+eval_data_count]
        train_ids = np.array(list(set(unique_ids) - set(val_ids)))

        train_df = df[df.id.isin(train_ids)]
        val_df = df[df.id.isin(val_ids)]

        # remobe q=qq and q=rr rows
        val_df = val_df[val_df.q != val_df.qq]
        val_df = val_df[val_df.r != val_df.rr]

        train_df.to_csv(
            os.path.join(train_folder, f"train_{i+1}.csv"), index=False)

        val_df.to_csv(
            os.path.join(val_folder, f"val_{i+1}.csv"), index=False)


if __name__ == "__main__":

    OFFICIAL_CSV = "dataset/official_data.csv"
    SAVE_TO = "dataset"
    SEED = 2022
    create_train_val(OFFICIAL_CSV, SAVE_TO, SEED)
