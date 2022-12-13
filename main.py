from ast import parse
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from model import MInterface
from data import DInterface
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val/epoch_score',
        mode='max',
        patience=3,
        min_delta=0.002
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='step'))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val/epoch_score',
        dirpath=f'./saved_models/{args.tag}/{args.exp_name}',
        filename='{epoch:02d}-[{val/epoch_score:.3f}]',
        save_top_k=1,
        mode='max',
        save_last=False,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True
    ))

    return callbacks


def main(args):

    if args.global_steps is None and args.checkpoint_path is None:

        # total train_data_counts = train data + extra data
        train_data_counts = len(pd.read_csv(args.train_csv_path).id.unique())

        if args.extra_train_file is not None:
            train_data_counts += len(pd.read_csv(args.extra_train_file).id.unique())

        args.global_steps = (train_data_counts // (args.batch_size *
                             args.accu_batches)) * args.max_epochs
        print(
            f'Global_steps is set to {args.global_steps}, Unique train data counts is {train_data_counts}')

        print(
            f"Batch_size {args.batch_size}, Accumulate_grad_batches {args.accu_batches}")

    if args.checkpoint_path is None:

        print("No checkpoint path provided. Training from scratch.")
        model = MInterface(**vars(args))
        data_module = DInterface(**vars(args))
        print("load model done")

    else:
        print(f"Loading checkpoint from {args.checkpoint_path}")

        model = MInterface.load_from_checkpoint(args.checkpoint_path,
                                                test_csv_path=args.test_csv_path,
                                                val_csv_path=args.val_csv_path,
                                                pred_csv=args.pred_csv,
                                                bin_threadhold=args.bin_threadhold,
                                                pred_weight=args.pred_weight)
        hps = model.hparams
        args.pretrained_tokenizer = hps.pretrained_tokenizer
        args.dataset = hps.dataset
        print(f"threadhold {hps.bin_threadhold}")

        data_module = DInterface(**vars(args))

        print("Test data loaded, dataset = ", args.dataset)

    # logger
    tb_logger = pl.loggers.TensorBoardLogger(
        name=f"{args.exp_name.replace('/', '_')}", save_dir=args.log_dir, default_hp_metric=False)

    trainer = Trainer.from_argparse_args(
        args, enable_progress_bar=False, callbacks=load_callbacks(), logger=tb_logger)

    if args.checkpoint_path is None:
        trainer.fit(model, datamodule=data_module)
    else:

        if args.test_csv_path is not None:
            print(f"Testing ... The result will be stored in {args.pred_csv}")
            trainer.test(model, data_module)

        if args.val_csv_path is not None:
            print(
                f"Validating ... The result will be stored in {args.pred_csv} and compared with {args.val_csv_path}")
            data_module.setup(stage='validation')
            val_loader = data_module.val_dataloader()
            trainer.validate(model, dataloaders=val_loader)


if __name__ == '__main__':

    parser = ArgumentParser()

    # experience name
    parser.add_argument('--exp_name', default='aicup', type=str)
    parser.add_argument('--tag', default='aicup', type=str)
    # Basic Training Control
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--accu_batches', default=1, type=int)

    parser.add_argument('--pretrained_model', type=str)
    parser.add_argument('--pretrained_tokenizer', type=str)
    parser.add_argument('--pretrained_config', type=str)

    # LR Scheduler
    parser.add_argument(
        '--lr_scheduler', choices=['step', 'cosine', 'cosine_warmup', 'linear_warmup', 'linear'], default='linear', type=str)

    parser.add_argument('--lr_warmup_ratio', default=0.1, type=float)
    parser.add_argument('--global_steps', default=None, type=int)

    # for step (stepLR) Scheduler, step on epoch
    parser.add_argument('--step_size', default=2, type=float)
    parser.add_argument('--gamma', default=0.5, type=float,
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

    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument(
        '--log_dir', default='lightning_logs', type=str)

    # Dataset and Model
    parser.add_argument('--dataset', default='v3_dataset', type=str)
    parser.add_argument('--model_name', default='v2_model', type=str)

    parser.add_argument('--extra_train_file', default=None, type=str)

    parser.add_argument(
        '--train_csv_path', default=None, type=str)
    parser.add_argument(
        '--val_csv_path', default=None, type=str)
    parser.add_argument(
        '--test_csv_path', default=None, type=str)

    # Model Hyperparameters
    parser.add_argument('--ce_weight', nargs='+', default=None, type=float)
    parser.add_argument('--label_smoothing', default=0.0, type=float)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str)

    parser.add_argument("--pred_csv", default=None, type=str)
    parser.add_argument("--pred_weight", nargs='+', default=None, type=float)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # prevent / in exp_name
    args.exp_name = args.exp_name.replace('/', '_')

    if args.pred_csv is None:
        if not os.path.exists(f"pred"):
            os.mkdir(f"pred")
        args.pred_csv = f"pred/{args.exp_name}.csv"

    if args.extra_train_file == "None":
        args.extra_train_file = None

    pl.seed_everything(args.seed, workers=True)
    main(args)
