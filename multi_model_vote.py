from ast import parse
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pandas as pd
from model import MInterface
from data import DInterface
import os
import torch
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "true"

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

MODEL_PATH = [ 
    # a list that contains the path of all models
    # for example:
    # "saved_models/deberta_None_train_val/Fold_1/03-[0.522].ckpt",
    # "saved_models/deberta_None_train_val/Fold_2/01-[0.512].ckpt",
]

def main(args):
    """
    Load all models and create test dataloder
    """
    model_list = []

    for model_path in MODEL_PATH:
        model = MInterface.load_from_checkpoint(model_path,
                                                ).to(DEVICE)

        model.eval()
        model_list.append(model)

    args.dataset = model_list[0].hparams.dataset
    args.max_seq_len = model_list[0].hparams.max_seq_len
    args.pretrained_tokenizer = model_list[0].hparams.pretrained_tokenizer

    print("args.dataset:", args.dataset)
    print("args.max_seq_len:", args.max_seq_len)
    print("args.pretrained_tokenizer:", args.pretrained_tokenizer)

    data_module = DInterface(**vars(args))
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()

    print(f"Test data loaded, using {data_module.dataset} dataset")
    print(f"Test file is {args.test_csv_path}")
    deberta_vote(args, model_list, test_loader)


def deberta_vote(args, model_list, test_loader):
    
    tokenizer = test_loader.dataset.tokenizer
    
    with torch.no_grad():

        total_preds = []

        for batch in tqdm(test_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(DEVICE)

            batch_size, seq_len = batch['input_ids'].shape
            vote_pred = torch.zeros(
                (batch_size, seq_len, 2), dtype=torch.float, device=DEVICE)

            # pred.shape = (batch_size, seq_len)
            # combine all models prediction
            for model in model_list:
                vote_pred += model(**batch).softmax(-1)

            vote_pred = vote_pred.argmax(dim=-1)

            # a_pred is a tensor of 0 and 1
            # a_pred.shape = (seq_len)
            for b_idx, a_pred in enumerate(vote_pred):

                valid = batch["valid"][b_idx]  # a tensor of 0 and 1
                input_ids = batch["input_ids"][b_idx]  # a tensor of ids

                # a list of str tokens
                q_nltk_tokens = batch["q_nltk_tokens"][b_idx]
                # a list of str tokens
                r_nltk_tokens = batch["r_nltk_tokens"][b_idx]

                # the following code is to recover the original tokens
                sep_idx = (input_ids == tokenizer.sep_token_id).nonzero(
                    as_tuple=True)[0][0].item()

                q_preds = a_pred[:sep_idx][valid[:sep_idx] == 1]
                r_preds = a_pred[sep_idx+1:][valid[sep_idx+1:] == 1]
                q = []
                r = []

                for i, pd_ in enumerate(q_preds):
                    if pd_ == 1:
                        q.append(q_nltk_tokens[i])
                        
                for i, pd_ in enumerate(r_preds):
                    if pd_ == 1:
                        r.append(r_nltk_tokens[i])

                q, r = " ".join(q), " ".join(r)
                a_sample_pred = {
                    # a id may have many seperate q and r
                    "id": batch["id"][b_idx],
                    "q": f"\"{q}\"",
                    "r": f"\"{r}\"",
                }

                assert isinstance(a_sample_pred["q"], str) and isinstance(
                    a_sample_pred["r"], str)
                total_preds.append(a_sample_pred)

        df = pd.DataFrame(total_preds, columns=["id", "q", "r"], dtype=str)
        df.to_csv(args.vote_output, index=False)
        print("Test done, save to", args.vote_output)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--vote_output', type=str,
                        default='vote.csv', help='output file name')
    # Basic Training Control
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--pretrained_tokenizer', type=str,
                        default=None)
    # Dataset and Model
    parser.add_argument('--dataset', default=None, type=str)

    parser.add_argument(
        '--test_csv_path', 
        type=str, 
        help='test csv file path')
    
    parser.add_argument('--max_seq_len', default=1024, type=int)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    main(args)
