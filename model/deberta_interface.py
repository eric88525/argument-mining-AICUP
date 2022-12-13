import inspect
from re import S
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from .common import overlap_rate, compare_two_csv
from collections import defaultdict, namedtuple
from transformers import AutoTokenizer
import pandas as pd
import math
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import os


class MInterface(pl.LightningModule):
    def __init__(self, model_name, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        self.model_name = model_name

        self.automatic_optimization = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_tokenizer)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def get_batch_loss(self, pred, labels):
        """
        pred: (batch_size, seq_len, class_num)
        labels: A len = batch_size List of tensor (N, seq_len)
        """
        batch_loss = 0

        batch_size, seq_len, class_num = pred.shape

        for b_idx in range(batch_size):

            a_pred = pred[b_idx]  # (seq_len, class_num)
            multi_label = labels[b_idx]  # N, seq_len
            loss = 0
            for i in range(multi_label.shape[0]):
                loss += self.loss_function(a_pred.view(-1, 2),
                                           multi_label[i].view(-1))
            loss /= multi_label.shape[0]
            batch_loss += loss

        return batch_loss / batch_size

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()

        # pred.shape = (batch size, seq_len, class_num)
        pred = self(**batch)
        batch_loss = self.get_batch_loss(pred, batch["labels"])
        self.manual_backward(batch_loss)

        if (batch_idx + 1) % self.hparams.accu_batches == 0:
            opt.step()
            opt.zero_grad()

        # step on epoch
        if self.hparams.lr_scheduler in ['step', 'cosine', 'linear']:
            if self.trainer.is_last_batch:
                sch = self.lr_schedulers()
                sch.step()
        else:  # step on batch
            if (batch_idx + 1) % self.hparams.accu_batches == 0:
                sch = self.lr_schedulers()
                sch.step()

        self.log('train/step_loss', batch_loss, on_step=True, on_epoch=False)

        return {'loss':  batch_loss}

    def training_epoch_end(self, output):
        # collect loss from all devices
        epcho_sum_loss = torch.stack([x['loss'] for x in output]).mean()
        self.log('train/epoch_loss', epcho_sum_loss,
                 on_epoch=True, on_step=False)

    def ids_to_qr(self, pred_ids):

        assert isinstance(pred_ids, torch.Tensor) and pred_ids.dim() == 1

        sep_idx = (pred_ids == self.tokenizer.sep_token_id).nonzero(
            as_tuple=True)[0][0].item()

        # print("sep ids", sep_idx)
        q = self.tokenizer.decode(pred_ids[:sep_idx], skip_special_tokens=True)
        r = self.tokenizer.decode(
            pred_ids[sep_idx+1:], skip_special_tokens=True)
        return q, r

    def validation_step(self, batch, batch_idx):

        # pred.shape = (batch_size, seq_len)
        pred = self(**batch).argmax(dim=-1)
        batch_preds = []

        # a_pred is a tensor of 0 and 1
        # a_pred.shape = (seq_len)
        for b_idx, a_pred in enumerate(pred):

            valid = batch["valid"][b_idx]  # a tensor of 0 and 1
            input_ids = batch["input_ids"][b_idx]  # a tensor of ids

            # a list of str tokens
            q_nltk_tokens = batch["q_nltk_tokens"][b_idx]
            # a list of str tokens
            r_nltk_tokens = batch["r_nltk_tokens"][b_idx]

            # the following code is to recover the original tokens
            sep_idx = (input_ids == self.tokenizer.sep_token_id).nonzero(
                as_tuple=True)[0][0].item()

            q_preds = a_pred[:sep_idx][valid[:sep_idx] == 1]
            r_preds = a_pred[sep_idx+1:][valid[sep_idx+1:] == 1]
            q = []
            r = []

            for i, pd in enumerate(q_preds):
                if pd == 1:
                    q.append(q_nltk_tokens[i])

            for i, pd in enumerate(r_preds):
                if pd == 1:
                    r.append(r_nltk_tokens[i])

            q, r = " ".join(q), " ".join(r)

            pred_sample = {
                "id": batch["id"][b_idx],
                "qq": q,
                "rr": r
            }
            batch_preds.append(pred_sample)


        return {'preds': batch_preds}

    def validation_epoch_end(self, output):

        preds = []  # list of dict {"id", "qq", "rr"}

        for batch in output:
            preds.extend(batch["preds"])

        df = pd.DataFrame(preds, columns=["id", "qq", "rr"])
        df.to_csv(self.hparams.pred_csv, index=False)

        score = compare_two_csv(pred_csv=self.hparams.pred_csv,
                                label_csv=self.hparams.val_csv_path)

        base, f_format = os.path.splitext(self.hparams.pred_csv)
        score_fname = f"{base}_[{score:.4f}]{f_format}"

        df.to_csv(score_fname, index=False)

        print("Save score to", score_fname)

        print(f"eval score: {score:.4f}")
        self.log("val/epoch_score", score, on_epoch=True)

    def test_step(self, batch, batch_idx):

        # pred.shape = (batch_size, seq_len)
        pred = self(**batch).argmax(dim=-1)
        batch_preds = []

        # a_pred is a tensor of 0 and 1
        # a_pred.shape = (seq_len)
        for b_idx, a_pred in enumerate(pred):

            valid = batch["valid"][b_idx]  # a tensor of 0 and 1
            input_ids = batch["input_ids"][b_idx]  # a tensor of ids

            # a list of str tokens
            q_nltk_tokens = batch["q_nltk_tokens"][b_idx]
            # a list of str tokens
            r_nltk_tokens = batch["r_nltk_tokens"][b_idx]

            # the following code is to recover the original tokens
            sep_idx = (input_ids == self.tokenizer.sep_token_id).nonzero(
                as_tuple=True)[0][0].item()

            q_preds = a_pred[:sep_idx][valid[:sep_idx] == 1]
            r_preds = a_pred[sep_idx+1:][valid[sep_idx+1:] == 1]
            q = []
            r = []

            for i, pd in enumerate(q_preds):
                if pd == 1:
                    q.append(q_nltk_tokens[i])

            for i, pd in enumerate(r_preds):
                if pd == 1:
                    r.append(r_nltk_tokens[i])

            q, r = " ".join(q), " ".join(r)

            pred_sample = {
                "id": batch["id"][b_idx],
                "qq": q,
                "rr": r
            }
            batch_preds.append(pred_sample)


        return {'preds': batch_preds}

    def test_epoch_end(self, output):

        preds = []  # list of dict {"id", "pred_qq", "pred_rr"}

        for batch in output:
            preds.extend(batch["preds"])

        df = pd.DataFrame(preds, columns=["id", "q", "r"], dtype=str)

        print("Test done, save to", self.hparams.pred_csv)
        df.to_csv(self.hparams.pred_csv, index=False)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(
                nd in name for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [param for name, param in param_optimizer if any(
                nd in name for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.hparams.lr)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.step_size,
                                       gamma=self.hparams.gamma)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.T_max,
                                                  eta_min=self.hparams.eta_min)
            elif self.hparams.lr_scheduler == 'linear':
                scheduler = lrs.LinearLR(
                    optimizer, total_iters=self.hparams.total_iters)

            elif self.hparams.lr_scheduler == 'cosine_warmup':
                scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=math.ceil(
                                                                self.hparams.lr_warmup_ratio * self.hparams.global_steps),
                                                            num_training_steps=self.hparams.global_steps)
            elif self.hparams.lr_scheduler == 'linear_warmup':

                print("linear warmup")
                print("warmup ratio", self.hparams.lr_warmup_ratio)
                print("global steps", self.hparams.global_steps)
                print("warmup steps", math.ceil(
                    self.hparams.lr_warmup_ratio * self.hparams.global_steps))

                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=math.ceil(
                                                                self.hparams.lr_warmup_ratio * self.hparams.global_steps),
                                                            num_training_steps=self.hparams.global_steps)
            else:
                raise ValueError('Invalid lr_scheduler type!')

        return [optimizer], [scheduler]

    def configure_loss(self):

        loss = self.hparams.loss.lower()

        if loss == 'mse':
            self.loss_function = nn.MSELoss()
        elif loss == 'l1':
            self.loss_function = nn.L1Loss()
        elif loss == 'bce':
            self.loss_function = nn.BCEWithLogitsLoss()
        elif loss == 'ce':
            if self.hparams.ce_weight is None:
                label_smoothing = 0.0
                if self.hparams.keys() & {'label_smoothing'}:
                    label_smoothing = self.hparams.label_smoothing
                self.loss_function = nn.CrossEntropyLoss(
                    reduction='mean', label_smoothing=label_smoothing)
            else:
                self.class_weight = torch.FloatTensor(
                    self.hparams.ce_weight, device=self.device)

                print("class_weight", self.class_weight)

                label_smoothing = 0.0
                if self.hparams.keys() & {'label_smoothing'}:
                    label_smoothing = self.hparams.label_smoothing

                self.loss_function = nn.CrossEntropyLoss(
                    weight=self.class_weight, reduction='mean', label_smoothing=label_smoothing)
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([_.capitalize() for _ in name.split('_')])
        try:

            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except Exception as E:
            print(Exception)
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')

        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
