import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
from torchvision.models import resnet50, vgg19, efficientnet_b3
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(description='CIFAR')
parser.add_argument('-b',
                    '--batch_size',
                    default=32,
                    type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e',
                    '--epochs',
                    default=30,
                    type=int,
                    help='number of total epochs (default: 30)')
parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
# Include DeepSpeed configuration arguments
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

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

if torch.distributed.get_rank() == 0:
    # cifar data is downloaded, indicate other ranks can proceed
    torch.distributed.barrier()

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transforms.Compose(
                                            [
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                                                    0.2023, 0.1994, 0.2010]),
                                            ]
                                        ))
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=2)


models = [resnet50, vgg19, efficientnet_b3]
result = {}
for model in models:
    net = model(num_classes=10)
    model_name = net.__class__.__name__

    parameters = filter(lambda p: p.requires_grad, net.parameters())

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=net, model_parameters=parameters, training_data=trainset)

    # fp16 = model_engine.fp16_enabled()
    # print(f'fp16={fp16}')

    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in tqdm(range(1)):
        model_engine.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader):

            if i * args.batch_size == 6400:
                break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
                model_engine.local_rank)
            # if fp16:
            #     inputs = inputs.half()
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

        model_engine.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # if fp16:
                #     images = images.half()
                outputs = net(images.to(model_engine.local_rank))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(
                    model_engine.local_rank)).sum().item()

    stop = time.time()
    result[str(model_name)] = (stop - start)

with open(f"result_deepspeed_{args.batch_size}.json", "w") as fp:
    json.dump(result, fp)
    