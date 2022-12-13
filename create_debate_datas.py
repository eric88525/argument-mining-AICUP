# %%

from datasets import Dataset, load_dataset, load_from_disk
import torch.nn as nn
from numpy.linalg import norm
from numpy import dot
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import csv
import pandas as pd
import os
import random

# Mean Pooling - Take attention mask into account for correct averaging


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class sbert():
    def __init__(self, device='cuda') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.model = AutoModel.from_pretrained(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)

    def get_embeddings(self, text_list: list, batch_size=16):

        embeddings = []

        print("Length of text_list: ", len(text_list))

        with torch.no_grad():
            for i in tqdm(range(0, len(text_list), batch_size)):
                batch = text_list[i:i+batch_size]
                encoded_input = self.tokenizer(
                    batch, padding=True, truncation=True, return_tensors='pt')

                for k in encoded_input:
                    encoded_input[k] = encoded_input[k].to(self.model.device)

                model_output = self.model(**encoded_input)
                sentence_embeddings = mean_pooling(
                    model_output, encoded_input['attention_mask']).cpu().numpy()

                embeddings.extend([se for se in sentence_embeddings])

        assert len(embeddings) == len(text_list)

        return embeddings


def create_emb_dataset(save_to, batch_size=16, max_words=300):

    print(f"{save_to}\n Bs {batch_size} w {max_words}")

    debate_sum = load_dataset("Hellisotherpeople/DebateSum")
    debate_sum = debate_sum["train"]

    # filder words <= max_words
    condition_ds = debate_sum.filter(
        lambda x: x["#WordsDocument"] <= max_words)

    docs = condition_ds["Full-Document"]
    extracts = condition_ds["Extract"]

    sbert_model = sbert()
    doc_embeddings = sbert_model.get_embeddings(
        docs, batch_size=batch_size)

    saved_docs = set()

    no_dup_doc_ext_emb = {
        "doc": [],
        "extract": [],
        "doc_emb": [],
    }

    for i in tqdm(range(len(docs))):
        if docs[i] not in saved_docs:
            saved_docs.add(docs[i])
            no_dup_doc_ext_emb["doc"].append(docs[i])
            no_dup_doc_ext_emb["extract"].append(extracts[i])
            no_dup_doc_ext_emb["doc_emb"].append(doc_embeddings[i])

    ds = Dataset.from_dict(no_dup_doc_ext_emb)
    ds.save_to_disk(save_to)

    print(f"Len {len(ds)}, Done")


def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))


def post_fix(x):
    return f"\"{x}\"".replace("\n", " ")


def to_n_fold(source_csv, output_folder, N=10):

    np.random.seed(2022)
    random.seed(2022)

    df = pd.read_csv(source_csv)
    unique_ids = df.id.unique()
    np.random.shuffle(unique_ids)

    eval_data_count = len(unique_ids) // N

    for i in range(N):

        valid_ids = unique_ids[i*eval_data_count:i *
                               eval_data_count+eval_data_count]
        valid_df = df[df.id.isin(valid_ids)]
        valid_df.to_csv(
            os.path.join(output_folder, f"extra_{i+1}.csv"), index=False)


def save_qr_pairs(dataset_path, save_to, threadhold, limit_N='max', device='cpu'):

    print(f"Matching... threadhold {threadhold} limit_N {limit_N} device {device}")

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    ds = load_from_disk(dataset_path)  # doc, extract, doc_emb

    N = len(ds) if limit_N == 'max' else limit_N
    print(f"N = {N}")

    pairs = []
    embs = torch.stack([torch.tensor(emb)
                       for emb in ds[:N]["doc_emb"]]).to(device)
    # the index that have been paired

    be_pair = set()
    mask_idx = []

    for r in tqdm(range(N)):

        if r in be_pair:
            continue

        row_sim = cos(embs[r], embs)  # N

        # mask itself
        row_sim[mask_idx] = -1
        row_sim[r] = -1

        sim_value, indices = torch.topk(row_sim, k=1)
        sim_value, indices = sim_value.item(), indices.item()

        if sim_value > threadhold and indices not in be_pair:
            # make to pair, no more matching
            be_pair.add(indices)
            be_pair.add(r)

            mask_idx.append(indices)
            mask_idx.append(indices)

            a_pair = {
                "id": len(pairs)+20000,
                "q": post_fix(ds[r]["doc"]),
                "qq": post_fix(ds[r]["extract"]),
                "r": post_fix(ds[indices]["doc"]),
                "rr": post_fix(ds[indices]["extract"]),
                "s": "AGREE"
            }

            pairs.append(a_pair)

    print(f"Len pairs: {len(pairs)}")
    df = pd.DataFrame(pairs)
    df.to_csv(save_to, index=False)


if __name__ == "__main__":

    # create embeddings
    WORK_DIR = os.getcwd()
    MAX_WORDS = 300
    THREADHOLD = 0.8

    if not os.path.isdir(os.sep.join([WORK_DIR, "dataset/embeds"])):
        os.makedirs(os.sep.join([WORK_DIR, "dataset/embeds"]))

    emb_path = os.sep.join(
        [WORK_DIR, "dataset/embeds", f"/doc_extract_emb_{MAX_WORDS}"])

    create_emb_dataset(save_to=emb_path, batch_size=32, max_words=MAX_WORDS)

    save_csv = os.sep.join(
        [WORK_DIR, f"dataset/len_{MAX_WORDS}_th_{THREADHOLD}.csv"])

    # create (q, r, AGREE) pairs
    save_qr_pairs(
        dataset_path=emb_path,
        save_to=save_csv,
        threadhold=THREADHOLD,
        limit_N='max',
        device='cuda'
    )

    output_folder = os.sep.join([
        WORK_DIR,
        "dataset",
        f"/len_{MAX_WORDS}_th_{THREADHOLD}"
    ])

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # split to 10 folds
    to_n_fold(source_csv=save_csv, output_folder=output_folder)
