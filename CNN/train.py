import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import wandb
from omegaconf import OmegaConf
from torchvision import transforms

import VGG
import loader
from utils import accuracy, AverageMeter

conf_list = OmegaConf.from_cli()["--CFG_PATH"]
FLAGS = OmegaConf.load(conf_list)

if FLAGS.WANDB:
    wandb_token = input("Wandb token: ")
    wandb.login(key=wandb_token)
    wandb.init(
        project="torch-model",
        config=dict(FLAGS),
        save_code=True,
    )


def main():
    cudnn.benchmark = True

    normalize = transforms.Normalize(
        mean=FLAGS.CIFAR10_MEAN,
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
        momentum=FLAGS.MOMENTUM,
        weight_decay=1e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        last_epoch=-1
    )

    best_prec1 = 0

    for epoch in range(FLAGS.MAX_EPOCH):
        print(f"Epoch {epoch + 1}")

        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        prec1 = validate(val_loader, model, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # modelのおもみの保存


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

        if i % FLAGS.PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    if FLAGS.WANDB:
        wandb.log(
            {
                "TRN Prec@1": top1.avg,
                "TRN CELoss": losses.avg
            }
        )


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.inference_mode():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % FLAGS.PRINT_FREQ == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    if FLAGS.WANDB:
        wandb.log(
            {
                "Val Prec@1": top1.avg,
                "Val CELoss": losses.avg
            }
        )

    return top1.avg


if __name__ == "__main__":
    main()
