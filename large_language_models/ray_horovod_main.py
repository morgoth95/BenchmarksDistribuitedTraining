import json
import os
import sys
import time
from pathlib import Path

import horovod.torch as hvd
import ray.train.torch
import torch
import torch.utils.data
import torch.distributed
from datasets import load_dataset
from filelock import FileLock
from ray.train import Trainer
from tqdm import tqdm
from transformers import GPT2TokenizerFast


os.environ["TRAIN_PLACEMENT_GROUP_TIMEOUT_S"] = "600"
ray.init(address="auto")


def save_to_file(dictionary, filename):
    save_path = Path(__file__).parent
    with open(save_path / filename, "w") as f:
        json.dump(dictionary, f)


def add_model_path():
    here = Path(__file__).parent
    working_dir = here / "gpt-2-Pytorch/GPT2"
    sys.path.append(str(working_dir))


def load_tokenized_dataset():
    wiki_full_dataset = load_dataset("wikipedia", "20220301.en")
    train_wiki_dataset = wiki_full_dataset["train"].shard(num_shards=800, index=0)
    wiki_datasets = train_wiki_dataset.train_test_split(test_size=0.1, shuffle=False)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    tokenized_datasets = wiki_datasets.map(lambda x: tokenizer(x['text']), batched=True)
    train_dataset = GPT2Dataset(tokenized_datasets["train"])
    test_dataset = GPT2Dataset(tokenized_datasets["test"])
    return train_dataset, test_dataset


def build_model():
    add_model_path()
    from config import GPT2Config
    from model import GPT2LMHeadModel

    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    return model


def get_batch(input_tensor):
    input_tensor = torch.tensor(input_tensor)
    rest = len(input_tensor) % 1024
    input_tensor = input_tensor[:len(input_tensor) - rest]
    if len(input_tensor) < 2048:
        return
    input_tensors = input_tensor.split(1024)
    output_tensors = input_tensors[1:]
    input_tensors = input_tensors[:-1]
    return list(zip(input_tensors, output_tensors))


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
        # if torch.cuda.is_available():
        #     input_tensor, output_tensor = input_tensor.cuda(), output_tensor.cuda()
        return input_tensor, output_tensor
        # return {"input_ids": input_tensor, "lm_labels": output_tensor}


def train_model(args_dict):
    args = args_dict["args"]
    seed = args.seed or 42
    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(seed)
    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    with FileLock(os.path.expanduser("~/.horovod_lock")):
        train_ds, test_ds = load_tokenized_dataset()
    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=hvd.size(), rank=hvd.rank()
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_ds, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler, **kwargs
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, sampler=test_sampler, **kwargs
    )
    model = build_model()
    if args.half:
        model = model.half()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate or 1e-3, eps=1e-4 if args.half else 1e-8
    )
    model.cuda()
    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )
    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    losses = []
    test_losses = []
    time_per_epoch = []
    for epoch in range(args.epochs):
        print(f"Running epoch {epoch + 1}/{args.epochs}")
        st = time.time()
        model.train()
        for i, data in tqdm(enumerate(train_dl)):
            loss = model(input_ids=data[0].cuda(), lm_labels=data[1].cuda())
            if args.track_training:
                losses.append(float(loss.cpu().detach().numpy()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Finished epoch {epoch+1}/{args.epochs}")
        test_loss = 0
        model.eval()
        print(f"Running test for epoch {epoch + 1}/{args.epochs}")
        for test_data in tqdm(test_dl):
            with torch.no_grad():
                loss = model(input_ids=test_data[0].cuda(), lm_labels=test_data[1].cuda())
                test_loss += float(loss.cpu().numpy())
        if len(test_dl) > 0:
            test_losses.append(test_loss / len(test_dl))
        time_per_epoch.append(time.time() - st)

    total_time = sum(time_per_epoch)
    print(f"Total time: {total_time}")
    return_dict = {
        "test": test_losses,
        "total_time": total_time,
        "time_per_epoch": time_per_epoch,
    }
    if args.track_training:
        return_dict["train"] = losses
    return return_dict


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-bs',
                        '--batch_size',
                        default=1,
                        type=int,
                        help='mini-batch size (default: 1)')
    parser.add_argument("--seed",
                        "-s",
                        default=42,
                        type=int,
                        help="Seed for fixing Horovod dataset split.")
    parser.add_argument('-e',
                        '--epochs',
                        default=10,
                        type=int,
                        help='number of total epochs (default: 10)')
    parser.add_argument('-lr',
                        '--learning_rate',
                        default=1e-3,
                        type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument("--half",
                        "-hf",
                        action="store_true",
                        help="Use half precision")
    parser.add_argument("--track_training",
                        "-tt",
                        action="store_true",
                        help="Track loss also while training.")
    parser.add_argument("--result_name",
                        type=str,
                        help="Filename where the result dictionary will be stored.")
    args = parser.parse_args()

    trainer = Trainer(backend="horovod", num_workers=4, use_gpu=True)

    trainer.start()
    results = trainer.run(train_model, config={"args": args})
    trainer.shutdown()
    result_name = args.result_name or f"result_ray_horovod_{args.batch_size}_{args.epochs}.json"
    if args.half:
        result_name = result_name.replace(".json", "_half.json")
    save_to_file(results, result_name)
