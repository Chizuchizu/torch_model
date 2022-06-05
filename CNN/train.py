import time
from typing import NamedTuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import transforms

import VGG
import loader


class FLAGS(NamedTuple):
    DEBUG = True
    BATCH_SIZE = 256
    # DATA_ROOT = './JAX-ResNet-CIFAR10/workspace/data/'
    LOG_ROOT = '.'
    MAX_EPOCH = 200  # ちゃんと回すときは200
    INIT_LR = 1e-1
    N_WORKERS = 12
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)


def main():
    cudnn.benchmark = True

    normalize = transforms.Normalize(
        mean=FLAGS.CIFAR100_MEAN,
        std=FLAGS.CIFAR10_STD
    )
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_loader, val_loader = loader.get_CIFAR10(
        train_transform=train_transform,
        val_transform=val_transform,
        train_batch_size=FLAGS.BATCH_SIZE,
        val_batch_size=FLAGS.BATCH_SIZE
    )

    criterion = nn.CrossEntropyLoss().cuda()
    model = VGG.vgg16()
    model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        FLAGS.INIT_LR,
        weight_decay=1e-4
    )

    for epoch in range(100):
        print(f"Epoch {epoch + 1}")

        train(train_loader, model, criterion, optimizer, epoch)


def train(
        train_loader, model, criterion, optimizer, epoch
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


main()
