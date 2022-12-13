# %%
from torch.utils.data import DataLoader
import math
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
from typing import List
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
try:
    from common import tokens_alignment
except:
    from .common import tokens_alignment
import re
from collections import OrderedDict
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast


def clean_text(text):

    text = text.strip('\"')

    sign = [r'``', r'\*', r"--"]
    sign_pattern = re.compile("|".join(sign))
    text = sign_pattern.sub("", text)

    to_replace = OrderedDict({
        r"\.{2,}": "",
        r"\(.*?http.*?\)": "",
        r"http.+?(?:html|htm|php|jpg|.com\/\s|pdf|tpl|gif|ph|asp|stm|ark|amp|[0-9]{3,}\/)": "",
    })

    for sign, replace_with in to_replace.items():
        text = re.sub(sign, replace_with, text)

    text = text.strip()
    return text


class DebertaV1(Dataset):
    def __init__(self, file_path, pretrained_tokenizer="microsoft/deberta-v3-base", max_seq_len: int = 1024,
                 extra_train_file=None, stage="train"):
        """
        Inits conlldataset

        Args:
            file_path: Path to the file
            pretrained_tokenizer: Pretrained model name e.g. bert-base-uncased or the path of tokenizer file
            max_seq_len: The max sequence length
            extra_train_file: Path to the extra train file
            stage: train, validation or test
        """
        self.tokenizer = DebertaV2TokenizerFast.from_pretrained(
            pretrained_tokenizer)
        self.useless = set("\"#$%&\'()*+ -/:;<=>@[\]^_\`{|}~")
        self.max_seq_len = max_seq_len
        self.stage = stage

        # [CLS] q [SEP] r [SEP]
        self.special_token_counts = 3

        # ['id', 'q', 'r', 's', 'qq', 'rr']
        self.extra_train_file = extra_train_file
        self.datas = self.read_file(file_path)

    def read_file(self, path):
        """Preprocess the dataframe"""
        df = pd.read_csv(path)

        # remove \" char from start and end
        if self.stage in ["train"]:
            for col in ['q', 'r', 'qq', 'rr']:
                df[col] = df[col].apply(clean_text)

        elif self.stage in ["validation", "test"]:
            for col in ['q', 'r']:
                df[col] = df[col].apply(clean_text)

        df["s"] = df["s"].map({"DISAGREE": 0, "AGREE": 1})

        if self.stage in ["train", "validation"]:
            df = df.groupby(["id", "s", "q", "r"]).aggregate(
                {
                    "qq": list,
                    "rr": list
                }
            ).reset_index()
        elif self.stage in ["test"]:
            df = df.groupby(["id", "s"]).aggregate(
                {
                    "q": lambda x: x.iloc[0],
                    "r": lambda x: x.iloc[0],
                }
            ).reset_index()

        datas = df.to_dict("records")

        # add extra train data
        if self.extra_train_file is not None and self.stage in ["train"]:
            extra_df = pd.read_csv(self.extra_train_file)

            extra_df["s"] = extra_df["s"].map({"DISAGREE": 0, "AGREE": 1})

            for col in ['q', 'r', 'qq', 'rr']:
                extra_df[col] = extra_df[col].apply(clean_text)

            extra_df = extra_df.groupby(["id", "s", "q", "r"]).aggregate(
                {
                    "qq": list,
                    "rr": list
                }
            ).reset_index()
            extra_datas = extra_df.to_dict("records")
            print("Extra train data size:", len(extra_datas))
            datas.extend(extra_datas)

        return datas

    def __len__(self):
        return len(self.datas)

    def get_input_ids_label_valid(self, text: str, argument: str):
        """
        Get input ids, labels, valid ids and nltk tokens

        """
        text_tokens = [w for w in word_tokenize(text) if w not in self.useless]
        argument_ids = [w for w in word_tokenize(
            argument) if w not in self.useless]
        text_labels = self.get_label(text_tokens, argument_ids)

        input_ids = []
        labels = []
        valid = []

        for t_token, t_label in zip(text_tokens, text_labels):
            ids = self.tokenizer.encode(
                t_token, add_special_tokens=False)

            input_ids.extend(ids)
            # only use the first token as prediction
            labels.extend([t_label] + [-100]*(len(ids)-1))
            valid.extend([1] + [0]*(len(ids)-1))

        return (input_ids, labels, valid, text_tokens)

    def get_input_ids_and_valid(self, text: str):

        text_tokens = [w for w in word_tokenize(text) if w not in self.useless]

        input_ids, valid = [], []

        for t_token in text_tokens:
            ids = self.tokenizer.encode(
                t_token, add_special_tokens=False)
            input_ids.extend(ids)
            valid.extend([1] + [0]*(len(ids)-1))

        return (input_ids, valid, text_tokens)

    def __getitem__(self, index):

        # train validation
        # key=['id', 'q', 'r', 's', 'qq', 'rr']
        # test
        # key=['id', 'q', 'r', 's']
        data = self.datas[index]
        # a q, r pair may have many qq, rr pairs
        multi_labels = []

        if self.stage in ["train", "validation"]:
            for qq, rr in zip(data["qq"], data["rr"]):

                q_token_ids, qq_label, q_ids_valid, q_nltk_tokens = self.get_input_ids_label_valid(
                    text=data["q"], argument=qq)
                r_token_ids, rr_label, r_ids_valid, r_nltk_tokens = self.get_input_ids_label_valid(
                    text=data["r"], argument=rr)

                multi_labels.append([qq_label, rr_label])

        elif self.stage in ["test"]:
            q_token_ids, q_ids_valid, q_nltk_tokens = self.get_input_ids_and_valid(
                text=data["q"])
            r_token_ids, r_ids_valid, r_nltk_tokens = self.get_input_ids_and_valid(
                text=data["r"])

        if len(q_token_ids) + len(r_token_ids) <= self.max_seq_len - self.special_token_counts:
            ...
        elif len(r_token_ids) <= self.max_seq_len // 2:  # [q1, q2, ... qn] [r]
            q_token_ids = q_token_ids[: self.max_seq_len -
                                      len(r_token_ids) - self.special_token_counts]
        elif len(q_token_ids) <= self.max_seq_len // 2:  # [q] [r1, r2, ... rn]
            r_token_ids = r_token_ids[: self.max_seq_len -
                                      len(q_token_ids) - self.special_token_counts]
        else:
            q_token_ids = q_token_ids[: (
                self.max_seq_len-self.special_token_counts) // 2]
            r_token_ids = r_token_ids[: (
                self.max_seq_len-self.special_token_counts) // 2]

        # after truncation, we need to update the valid
        q_ids_valid = q_ids_valid[: len(q_token_ids)]
        r_ids_valid = r_ids_valid[: len(r_token_ids)]
        valid = [0] + q_ids_valid + [0] + r_ids_valid + [0]

        #  [CLS] A [SEP] B [SEP]
        input_ids = self.tokenizer.build_inputs_with_special_tokens(
            q_token_ids, r_token_ids)

        assert len(input_ids) <= self.max_seq_len and len(
            valid) == len(input_ids)

        attention_mask = [1 for _ in range(len(input_ids))]
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(
            q_token_ids, r_token_ids)

        sample = {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids),
            "id": data["id"],
            "q": data["q"],
            "r": data["r"],
            "valid": torch.LongTensor(valid),
            "q_nltk_tokens": q_nltk_tokens,
            "r_nltk_tokens": r_nltk_tokens,
        }

        labels = []

        if self.stage in ["train", "validation"]:
            # truncate the labels

            for q_label, r_label in multi_labels:

                q_label = q_label[: len(q_token_ids)]
                r_label = r_label[: len(r_token_ids)]

                label = torch.LongTensor(
                    [data["s"]] + q_label + [-100] + r_label + [-100])

                labels.append(label)

                assert len(input_ids) == len(label)

            sample["qq"] = data["qq"]
            sample["rr"] = data["rr"]

            sample["labels"] = torch.stack(labels)

        return sample

    def get_label(self, seq, sub_seq) -> List[int]:
        """Get valid token ids

        Args:
            token_ids: The token ids
            alignments: The alignments, for example [[0,1], [1,2], [3, -100]]
                        first element is subsequence index, second element is sequence index

        Returns:
            The valid token ids.
            For example: [0, 0, 1, 1, 1]
            1 is valid token, 0 is not valid token
        """
        alignments = tokens_alignment(seq=seq, sub_seq=sub_seq)
        label = [0 for _ in range(len(seq))]

        for _, seq_idx in alignments:
            if seq_idx >= 0:
                label[seq_idx] = 1

        return label

    def collate_fn(self, batch):

        max_len = max([len(x["input_ids"]) for x in batch])

        for b_idx in range(len(batch)):

            pad_len = max_len - len(batch[b_idx]["input_ids"])

            batch[b_idx]["input_ids"] = F.pad(
                batch[b_idx]["input_ids"], (0, pad_len), value=self.tokenizer.pad_token_id)

            batch[b_idx]["attention_mask"] = F.pad(
                batch[b_idx]["attention_mask"], (0, pad_len), value=0)

            batch[b_idx]["token_type_ids"] = F.pad(
                batch[b_idx]["token_type_ids"], (0, pad_len), value=0)

            batch[b_idx]["valid"] = F.pad(
                batch[b_idx]["valid"], (0, pad_len), value=0)

            if self.stage in ["train", "validation"]:
                batch[b_idx]["labels"] = F.pad(
                    batch[b_idx]["labels"], (0, pad_len), value=-100)

        return_batch = {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "token_type_ids": torch.stack([x["token_type_ids"] for x in batch]),
            "id": [x["id"] for x in batch],
            "q": [x["q"] for x in batch],
            "r": [x["r"] for x in batch],
            "valid": torch.stack([x["valid"] for x in batch]),
            "q_nltk_tokens": [x["q_nltk_tokens"] for x in batch],
            "r_nltk_tokens": [x["r_nltk_tokens"] for x in batch],
        }

        if self.stage in ["train", "validation"]:
            return_batch["labels"] = [x["labels"] for x in batch]

        return return_batch