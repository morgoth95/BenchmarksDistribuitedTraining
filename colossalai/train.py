import colossalai

# ./config.py refers to the config file we just created in step 1
colossalai.launch_from_torch(config='./config.py')

from pathlib import Path
from colossalai.logging import get_dist_logger
import torch
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50, vgg19, efficientnet_b3
from tqdm import tqdm
import time

models = [resnet50, vgg19, efficientnet_b3]

# build logger
logger = get_dist_logger()

result = {}

for model in models:
    model = model(num_classes=10)

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

    # build dataloaders
    train_dataloader = get_dataloader(dataset=train_dataset,
                                    shuffle=True,
                                    batch_size=gpc.config.BATCH_SIZE,
                                    num_workers=1,
                                    pin_memory=True,
                                    )

    test_dataloader = get_dataloader(dataset=test_dataset,
                                    add_sampler=False,
                                    batch_size=gpc.config.BATCH_SIZE,
                                    num_workers=1,
                                    pin_memory=True,
                                    )

    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # lr_scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                        optimizer,
                                                                        criterion,
                                                                        train_dataloader,
                                                                        test_dataloader,
                                                                        )

    start = time.time()
    for epoch in tqdm(range(gpc.config.NUM_EPOCHS)):
        # execute a training iteration
        engine.train()
        for i, (img, label) in enumerate(train_dataloader):

            # Train only on 6400 images to reduce time
            if i * gpc.config.BATCH_SIZE == 6400:
                break

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
        
        # execute a testing iteration
        engine.eval()
        correct = 0
        total = 0
        for img, label in test_dataloader:
            img = img.cuda()
            label = label.cuda()
            
            # run prediction without back-propagation
            with torch.no_grad():
                output = engine(img)
                test_loss = engine.criterion(output, label)
            
            # compute the number of correct prediction
            pred = torch.argmax(output, dim=-1)
            correct += torch.sum(pred == label)
            total += img.size(0)

    stop = time.time()

    import json

    result[str(model.__class__.__name__ )] = (stop - start)

with open(f"result_colossalai_{gpc.config.BATCH_SIZE}.json", "w") as fp:
    json.dump(result, fp)

