import time

import deepspeed
import torch
import torch.utils.data
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from baseline_main import build_model, save_to_file, get_batch


def load_tokenized_dataset():
    wiki_full_dataset = load_dataset("wikipedia", "20220301.en")
    train_wiki_dataset = wiki_full_dataset["train"].shard(num_shards=800, index=0)
    wiki_datasets = train_wiki_dataset.train_test_split(test_size=0.1)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    tokenized_datasets = wiki_datasets.map(lambda x: tokenizer(x['text']), batched=True)
    train_dataset = GPT2Dataset(tokenized_datasets["train"])
    test_dataset = GPT2Dataset(tokenized_datasets["test"])
    return train_dataset, test_dataset


class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_data):
        super().__init__()
        self._data = []
        for tokens in hf_data["input_ids"]:
            batch = get_batch(tokens)
            if batch is not None:
                self._data.extend(batch)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        input_tensor, output_tensor = self._data[item]
        if torch.cuda.is_available():
            input_tensor, output_tensor = input_tensor, output_tensor
        return {"input_ids": input_tensor, "lm_labels": output_tensor}


def train_model(args):
    model = build_model()
    if torch.distributed.get_rank() != 0:
        # might be downloading cifar data, let rank 0 download first
        torch.distributed.barrier()
    train_ds, test_ds = load_tokenized_dataset()
    if torch.distributed.get_rank() == 0:
        # cifar data is downloaded, indicate other ranks can proceed
        torch.distributed.barrier()
    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    # test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)
    model_engine, optimizer, train_dl, __ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_ds,
    )

    # losses = []
    # test_losses = []
    time_per_epoch = []
    for epoch in range(args.epochs):
        print(f"Running epoch {epoch + 1}/{args.epochs}")
        st = time.time()
        for data in tqdm(train_dl):
            data = {k: v.to(model_engine.local_rank) for k, v in data.items()}
            loss = model_engine(**data)
            # if args.track_training:
            #     losses.append(float(loss.cpu().detach().numpy()))
            model_engine.backward(loss)
            model_engine.step()

        # test_loss = 0
        # model_engine.eval()
        # print(f"Running test for epoch {epoch + 1}/{args.epochs}")
        # for test_data in tqdm(test_dl):
        #     test_data = {k: v.to(model_engine.local_rank) for k, v in test_data.items()}
        #     with torch.no_grad():
        #         loss = model_engine(**test_data)
        #         test_loss += float(loss.cpu().numpy())
        # if len(test_dl) > 0:
        #     test_losses.append(test_loss / len(test_dl))
        time_per_epoch.append(time.time() - st)

    total_time = sum(time_per_epoch)
    return_dict = {
    #    "test": test_losses,
        "total_time": total_time,
        "time_per_epoch": time_per_epoch,
    }
    # if args.track_training:
    #     return_dict["train"] = losses
    return return_dict


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-bs',
                        '--batch_size',
                        default=1,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=10,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument("--track_training",
                        "-tt",
                        action="store_true",
                        help="Track loss also while training.")
    parser.add_argument("--result_name",
                        type=str,
                        help="Filename where the result dictionary will be stored.")
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    deepspeed.init_distributed()

    result_dict = train_model(args)
    result_name = args.result_name or f"result_deepspeed_{args.train_batch_size}_{args.epochs}.json"
    save_to_file(result_dict, result_name)
