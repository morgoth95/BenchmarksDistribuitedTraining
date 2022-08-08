import time

import deepspeed
import torch
import torch.utils.data
from tqdm import tqdm

from baseline_main import build_model, load_tokenized_dataset, save_to_file


def train_model(args):
    model = build_model()
    train_ds, test_ds = load_tokenized_dataset()
    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.test_batch_size)
    model_engine, optimizer, train_dl, __ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_ds,
    )

    losses = []
    test_losses = []
    time_per_epoch = []
    for epoch in range(args.epochs):
        print(f"Running epoch {epoch + 1}/{args.epochs}")
        st = time.time()
        for data in tqdm(train_dl):
            loss = model_engine(**data)
            if args.track_training:
                losses.append(float(loss.cpu().detach().numpy()))
            model_engine.backward(loss)
            model_engine.step()

        test_loss = 0
        model_engine.eval()
        print(f"Running test for epoch {epoch + 1}/{args.epochs}")
        for test_data in tqdm(test_dl):
            with torch.no_grad():
                loss = model_engine(**test_data)
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
    if args.track_training:
        return_dict["train"] = losses
    return return_dict


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-bs',
                        '--batch_size',
                        default=32,
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
    parser.add_argument("--test_batch_size",
                        "-tbs",
                        default=32,
                        type=int,
                        help="Batch size to be used for test.")
    parser.add_argument("--track_training",
                        "-tt",
                        action="return_true",
                        help="Track loss also while training.")
    parser.add_argument("--result_name",
                        type=str,
                        help="Filename where the result dictionary will be stored.")
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    deepspeed.init_distributed()
    if torch.distributed.get_rank() != 0:
        # might be downloading cifar data, let rank 0 download first
        torch.distributed.barrier()

    result_dict = train_model(args)
    result_name = args.result_name or f"result_deepspeed_{args.batch_size}_{args.epochs}.json"
    save_to_file(result_dict, result_name)
