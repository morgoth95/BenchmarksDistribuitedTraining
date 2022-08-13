import colossalai

# ./config.py refers to the config file we just created in step 1
colossalai.launch_from_torch(config='./config.py')

from pathlib import Path
import torch
import torch.utils.data as data_utils
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from colossalai.zero.init_ctx import ZeroInitContext
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR10
from torchvision.models import resnet152, swin_b, vit_l_16, vit_h_14
from tqdm import tqdm
import time
import json
import argparse



def prepare_data(args):
    # build datasets
    train_dataset = CIFAR10(
        root='./data',
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                    0.2023, 0.1994, 0.2010]),
            ]
        )
    )

    # Limit dataset size
    if args.limit_data:
        indices = torch.arange(args.limit_data)
        train_dataset = data_utils.Subset(train_dataset, indices)

    test_dataset = CIFAR10(
        root='./data',
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                    0.2023, 0.1994, 0.2010]),
            ]
        )
    )

    return train_dataset, test_dataset


def train_with_batch(model, train_dataset, test_dataset, batch_size, found_limit, result, args):
    if args.use_zero:
        with ZeroInitContext(target_device=torch.cuda.current_device(),
                        shard_strategy=gpc.config.zero.model_config.shard_strategy,
                        shard_param=True):
            model = model(num_classes=10)
    else:
        model = model(num_classes=10)

    model_name = model.__class__.__name__

    # build dataloaders
    train_dataloader = get_dataloader(dataset=train_dataset,
                                    shuffle=True,
                                    batch_size=batch_size,
                                    num_workers=1,
                                    pin_memory=True,
                                    )

    test_dataloader = get_dataloader(dataset=test_dataset,
                                    add_sampler=False,
                                    batch_size=batch_size,
                                    num_workers=1,
                                    pin_memory=True,
                                    )

    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.8, 0.999], eps=1e-8, weight_decay=3e-7)

    # lr_scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                        optimizer,
                                                                        criterion,
                                                                        train_dataloader,
                                                                        test_dataloader,
                                                                        )

    print(f"Using model {model_name} with bs {batch_size}")

    start = time.time()
    for epoch in tqdm(range(gpc.config.NUM_EPOCHS)):
        # execute a training iteration
        engine.train()
        for i, (img, label) in tqdm(enumerate(train_dataloader)):
            img = img.cuda()
            label = label.cuda()

            # set gradients to zero
            engine.zero_grad()

            # run forward pass
            output = engine(img)

            # compute loss value and run backward pass
            train_loss = engine.criterion(output, label)
            engine.backward(train_loss)

            # update parameters
            engine.step()

        # update learning rate
        lr_scheduler.step()

    stop = time.time()
    result[str(model_name)] = {"bs": batch_size, "time": (stop - start)}

    with open(f"result_colossalai_{args.name}.json", "w") as fp:
        json.dump(result, fp)

    if not found_limit:
        batch_size += 16
    else:
        batch_size += 4

    return batch_size


def train_model(args, model, train_dataset, test_dataset, result):

    batch_size = gpc.config.BATCH_SIZE

    found_limit = False

    while True:
        try:
            batch_size = train_with_batch(model, train_dataset, test_dataset, batch_size, found_limit, result, args)
        except Exception as e:
            print(e)
            if not found_limit:
                print(f"FOUND LIMIT BS: {batch_size}")
                batch_size -= 12
                found_limit = True
            else:
                print(f"OPTIMAL BS: {batch_size-4}")
                torch.cuda.empty_cache()
                break
        if not args.find_bs:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR')

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
    parser.add_argument('--use_zero',
                        action='store_true',
                        help='use zero optimization')

    args = parser.parse_args()

    trainset, testset = prepare_data(args)

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
        train_model(args, model, trainset, testset, result)

    # with open(f"result_colossalai_{args.name}.json", "w") as fp:
    #     json.dump(result, fp)
