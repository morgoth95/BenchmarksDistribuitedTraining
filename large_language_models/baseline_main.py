import json
import sys
import time
from pathlib import Path

import torch
import torch.utils.data
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm


def add_model_path():
    here = Path(__file__).parent
    working_dir = here / "gpt-2-Pytorch/GPT2"
    sys.path.append(working_dir)


def load_tokenized_dataset():
    wiki_full_dataset = load_dataset("wikipedia", "20220301.en")
    train_wiki_dataset = wiki_full_dataset["train"].shard(num_shards=800, index=0)
    wiki_datasets = train_wiki_dataset.train_test_split(test_size=0.1)
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
        if torch.cuda.is_available():
            input_tensor, output_tensor = input_tensor.cuda(), output_tensor.cuda()
        return {"input_ids": input_tensor, "lm_labels": output_tensor}


def train_model(max_epochs: int, lr: float, batch_size: int, test_batch_size: int, track_training_metrics: bool):
    model = build_model()
    train_ds, test_ds = load_tokenized_dataset()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size)

    losses = []
    test_losses = []
    time_per_epoch = []
    for epoch in range(max_epochs):
        print(f"Running epoch {epoch + 1}/{max_epochs}")
        model.train()
        st = time.time()
        for data in tqdm(train_dl):
            loss = model(**data)
            if track_training_metrics:
                losses.append(float(loss.cpu().detach().numpy()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        test_loss = 0
        print(f"Running test for epoch {epoch + 1}/{max_epochs}")
        model.eval()
        for test_data in tqdm(test_dl):
            with torch.no_grad():
                loss = model(**test_data)
                test_loss += float(loss.cpu().numpy())
        if len(test_dl) > 0:
            test_losses.append(test_loss / len(test_dl))
        time_per_epoch.append(time.time() - st)

    total_time = sum(time_per_epoch)
    return_dict = {
        "test": test_losses,
        "total_time": total_time,
        "time_per_epoch": time_per_epoch,
    }
    if track_training_metrics:
        return_dict["train"] = losses
    return return_dict


def save_to_file(dictionary, filename):
    save_path = Path(__file__).parent
    with open(save_path / filename, "w") as f:
        json.dump(dictionary, f)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--batch_size", "-bs", type=int, help="Batch size to be used for training.")
    parser.add_argument("--test_batch_size", "-tbs", type=int, help="Batch size to be used for test.")
    parser.add_argument("--epochs", "-e", type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", "-lr", type=float, help="The learning rate.")
    parser.add_argument("--track_training", "-tt", action="store_true", help="Track loss also while training.")
    parser.add_argument("--result_name", type=str, help="Filename where the result dictionary will be stored.")
    args = parser.parse_args()
    bs = args.batch_size or 1
    tbs = args.test_batch_size or bs
    epochs = args.epochs or 10
    learning_rate = args.learning_rate or 1e-3
    track_training = args.track_training or False
    result_name = args.result_name or f"result_{bs}_{epochs}.json"

    result_dict = train_model(epochs, learning_rate, bs, tbs, track_training)
    save_to_file(result_dict, result_name)
