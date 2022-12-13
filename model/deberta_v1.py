from transformers import AutoConfig, AutoModelForTokenClassification
import torch.nn as nn


class DebertaV1(nn.Module):
    def __init__(self, pretrained_model, **kwargs):
        super(DebertaV1, self).__init__()

        self.LM = AutoModelForTokenClassification.from_pretrained(
            pretrained_model,
            num_labels=2
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        return self.LM(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
