import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
from torchvision.models import resnet152, swin_b, vit_l_16, vit_h_14
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import datetime
import numpy as np
import time
from tqdm import tqdm
import json


def prepare_data(args):
    deepspeed.init_distributed()

    if torch.distributed.get_rank() != 0:
        # might be downloading cifar data, let rank 0 download first
        torch.distributed.barrier()

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transforms.Compose(
                                                [
                                                    transforms.Resize((224, 224)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                                                        0.2023, 0.1994, 0.2010]),
                                                ]
                                            ))

    # Limit dataset size
    if args.limit_data:
        indices = torch.arange(args.limit_data)
        trainset = data_utils.Subset(trainset, indices)

    if torch.distributed.get_rank() == 0:
        # cifar data is downloaded, indicate other ranks can proceed
        torch.distributed.barrier()

    return trainset


def train_with_batch(model, trainset, config, bs, found_limit, result, args):
    config["train_micro_batch_size_per_gpu"] = bs
    config["train_batch_size"] = bs * 4

    net = model(num_classes=10)
    model_name = net.__class__.__name__

    parameters = filter(lambda p: p.requires_grad, net.parameters())

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=net, model_parameters=parameters, training_data=trainset, config_params=config)

    fp16 = model_engine.fp16_enabled()
    print(f'fp16={fp16}')

    criterion = nn.CrossEntropyLoss()

    print(f"Using model {model_name} with bs {bs}")

    start = time.time()
    for epoch in tqdm(range(args.epochs)):  # loop over the dataset multiple times
        model_engine.train()
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
                model_engine.local_rank)
            if fp16:
                inputs = inputs.half()

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

    stop = time.time()
    result[str(model_name)] = {"bs": bs, "time": (stop - start)}

    with open(f"result_deepspeed_{args.name}.json", "w") as fp:
        json.dump(result, fp)

    if not found_limit:
        bs += 16
    else:
        bs += 4

    return bs


def train_model(args, model, trainset, result):
    with open(args.deepspeed_config, "r") as fp:
        config = json.load(fp)

    batch_size = config["train_micro_batch_size_per_gpu"]

    found_limit = False

    while True:
        try:
            batch_size = train_with_batch(model, trainset, config, batch_size, found_limit, result, args)
        except Exception as e:
            print(e)
            if not found_limit:
                print(f"FOUND LIMIT BS: {batch_size}")
                batch_size -= 12
                found_limit = True
            else:
                print(f"OPTIMAL BS: {batch_size - 4}")
                torch.cuda.empty_cache()
                break

        # End loop if using fixed batch size
        if not args.find_bs:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR')

    parser.add_argument('-e',
                        '--epochs',
                        default=1,
                        type=int,
                        help='number of total epochs (default: 1)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--name',
                        type=str,
                        default="",
                        help='name to identify the experiment result')
    parser.add_argument('--limit_data',
                        type=int,
                        default=None,
                        help='limit the number of data used to train the model')
    parser.add_argument('--models',
                        type=str,
                        help='delimited list input')
    parser.add_argument('--find_bs',
                        action='store_true',
                        help='find optimal bs per model')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    trainset = prepare_data(args)

    model_list = [model for model in args.models.split(',')]

    supported_models = {
        "resnet152": resnet152,
        "swin_b": swin_b,
        "vit_l_16": vit_l_16,
        "vit_h_14": vit_h_14,
    }

    models = [supported_models[key] for key in model_list]

    result = {}

    for model in models:
        train_model(args, model, trainset, result)

    # with open(f"result_deepspeed_{args.name}.json", "w") as fp:
    #     json.dump(result, fp)
