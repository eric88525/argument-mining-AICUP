from ast import parse
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from model import MInterface
from data import DInterface
import os
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import pandas as pd
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_callbacks():
    callbacks = []
    callbacks.append(
        TuneReportCallback(
            {
                "val/epoch_score": "val/epoch_score"
            },
            on="validation_end")
    )

    callbacks.append(plc.LearningRateMonitor(
        logging_interval='epoch'))

    return callbacks


def train_model_tune(config, tune_num_epochs):

    if config["global_steps"] is None:
        train_data_counts = len(pd.read_csv(
            config['train_csv_path']).id.unique())

        print("train_data_counts: ", train_data_counts)

        if config['extra_train_file'] is not None:
            train_data_counts += len(pd.read_csv(
                config['extra_train_file']).id.unique())

        config['global_steps'] = int(
            (train_data_counts // (config['batch_size'] * config['accu_batches'])) * tune_num_epochs)

        assert isinstance(config['global_steps'], int)

        print(
            f"Global_steps {config['global_steps']}, Unique train data counts is {train_data_counts}")
        print(
            f"Batch_size {config['batch_size']}, accu_batches {config['accu_batches']}")

    data_module = DInterface(**config)
    model = MInterface(**config)

    trainer = Trainer(
        max_epochs=tune_num_epochs,
        gpus=config["gpus"],
        accelerator='gpu',
        logger=TensorBoardLogger(
            save_dir=os.getcwd(),
            name="",
            version="."
        ),
        precision=16,
        callbacks=load_callbacks(),
        enable_progress_bar=False
    )

    trainer.fit(model, data_module)


def tune_asha(args):

    # edit this to your own search space
    search_config = {
        "batch_size": tune.choice([8, 4, 2]),
        "lr": tune.loguniform(5e-5, 3e-4),
        "ce_weight": tune.choice([
            [0.6999, 1.755],  # [0.1, 0.9] , [0.5, 0.5]
        ]),
        "lr_warmup_ratio": 0.1,
        "accu_batches": 4
    }

    config = vars(args)
    config.update(search_config)

    scheduler = ASHAScheduler(
        max_t=args.tune_num_epochs,
        grace_period=3,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=[k for k in search_config.keys()],
        metric_columns=["val/epoch_score"])

    train_fn_with_parameters = tune.with_parameters(train_model_tune,
                                                    tune_num_epochs=args.tune_num_epochs)
    resources_per_trial = {"cpu": 8, "gpu": 1}

    analysis = tune.run(train_fn_with_parameters,
                        resources_per_trial=resources_per_trial,
                        metric="val/epoch_score",
                        mode="max",
                        config=config,
                        num_samples=args.tune_num_samples,
                        scheduler=scheduler,
                        progress_reporter=reporter,
                        name=args.exp_name,
                        fail_fast=True,
                        )

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == '__main__':

    parser = ArgumentParser()

    # experience name
    parser.add_argument('--exp_name', default='test', type=str)

    # Basic Training Control
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=87, type=int)

    parser.add_argument('--pretrained_model', type=str)
    parser.add_argument('--pretrained_tokenizer', type=str)
    parser.add_argument('--pretrained_config', type=str)

    # LR Scheduler
    parser.add_argument(
        '--lr_scheduler', choices=['step', 'cosine', 'cosine_warmup', 'linear_warmup', 'linear'],
        default='linear_warmup', type=str)

    parser.add_argument('--global_steps', default=None, type=int)

    # for step (stepLR) Scheduler, step on epoch
    parser.add_argument('--step_size', default=2, type=float)
    parser.add_argument('--gamma', default=0.5, type=int,
                        help='lr *= gamma when epoch % step_size == 0')

    # for cosine (CosineAnnealingLR) Scheduler, step on epoch
    parser.add_argument('--T_max', default=5, type=float)
    parser.add_argument('--eta_min', default=1e-5, type=float)

    # for cosine_warmup (get_cosine_schedule_with_warmup) Scheduler, step on step
    parser.add_argument('--num_warmup_steps', default=100, type=int)
    parser.add_argument('--num_training_steps', default=1000, type=int)

    # for linear Scheduler, step on epoch
    parser.add_argument('--total_iters', default=3, type=int)

    # Training Info
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--label_smoothing', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    # Dataset and Model
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--extra_train_file', default=None, type=str)

    parser.add_argument(
        '--train_csv_path', default=None, type=str)
    parser.add_argument(
        '--val_csv_path', default=None, type=str)
    parser.add_argument(
        '--test_csv_path', default=None, type=str)

    # Model Hyperparameters
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument("--pred_csv", default=None, type=str)

    parser.add_argument('--tune_num_samples', default=20,
                        type=int, help='number of samples to tune')
    parser.add_argument('--tune_num_epochs', default=10,
                        type=int, help='number of epochs to tune')

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    if args.pred_csv is None:
        args.pred_csv = f"pred/{args.exp_name}.csv"

    if args.extra_train_file == "None":
        args.extra_train_file = None

    pl.seed_everything(args.seed)

    tune_asha(args)