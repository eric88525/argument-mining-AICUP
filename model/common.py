# %%
import torch.nn as nn
from functools import lru_cache
from typing import List
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from tqdm import tqdm
nltk.download('punkt')


def _init_weights(module, initializer_range=0.02):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(
            mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def LCS(text1: str, text2: str):
    """ Longest Common Subsequence"""
    m = len(text1)
    n = len(text2)
    t = [[0]*(n+1) for i in range(m+1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                t[i][j] = t[i-1][j-1] + 1
            else:
                t[i][j] = max(t[i][j - 1], t[i - 1][j])

    return t[m][n]


def overlap_rate(pred_str: str, label_str: str) -> float:
    """Get overlap between pred and label"""

    try:
        assert isinstance(pred_str, str) and isinstance(label_str, str)
    except AssertionError:
        print("pred_str: ", pred_str)
        print("label_str: ", label_str)
        raise AssertionError

    useless = set("!\"#$%&\'()*+, -./:;<=>?@[\]^_\`{|}~")

    pred_tokens = [t for t in word_tokenize(str(pred_str)) if t not in useless]
    label_tokens = [t for t in word_tokenize(
        str(label_str)) if t not in useless]

    lcs = LCS(pred_tokens, label_tokens)

    return lcs / (len(pred_tokens) + len(label_tokens) - lcs + 1e-8)


def compare_two_csv(pred_csv, label_csv) -> float:
    """Compare two csv file

    The pred_csv should have three columns: id, qq, rr
    qq and rr is the pred of given q and r

    The label_csv should have three columns: id, qq, rr
    qq and rr may have multiple pairs
    """
    pred_df = pd.read_csv(pred_csv).set_index("id")  # col = [id, qq, rr]
    pred_df.fillna("", inplace=True)  # fill nan with empty string

    label_df = pd.read_csv(label_csv)  # col = [id, qq, rr]
    label_df.fillna("", inplace=True)  # fill nan with empty string

    label_df = label_df.groupby(["id"]).aggregate(
        {"qq": list,
         "rr": list}).reset_index().set_index("id")

    if set(pred_df.index) != set(label_df.index):
        print("Index not fully match")

    common_index = set(pred_df.index).intersection(set(label_df.index))

    print("Total number of samples: ", len(common_index))

    if len(common_index) == 0:
        print("No common index")
        return 0

    scores = []

    for idx in common_index:

        pred_qq = pred_df.loc[idx].qq  # qq (str)
        pred_rr = pred_df.loc[idx].rr  # rr (str)

        label_qq_list = label_df.loc[idx].qq  # list of qq
        label_rr_list = label_df.loc[idx].rr  # list of rr

        max_score = 0.0

        for label_qq, label_rr in zip(label_qq_list, label_rr_list):
            score_ = overlap_rate(pred_qq, label_qq) + \
                overlap_rate(pred_rr, label_rr)
            max_score = max(max_score, score_)

            scores.append(max_score)

    score = sum(scores) / (2 * len(scores) + 1e-8)

    return score
