import argparse
import os
import shutil
import time
from tkinter import image_names

import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import wandb
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

from AlexNet import localizer_alexnet, localizer_alexnet_robust
from voc_dataset import *
from utils import *

USE_WANDB = True  # use flags, wandb is not convenient for debugging
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=1,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=4,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)') #256 default
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_false',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.to(device)

    for name, param in model.features.named_parameters():
        param.requires_grad = False

    # TODO (Q1.1): define loss function (criterion) and optimizer from [1]
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.001, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma= 0.1)



    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    # TODO (Q1.1): Create Datasets and Dataloaders using VOCDataset
    # Ensure that the sizes are 512x512
    # Also ensure that data directories are correct
    # The ones use for testing by TAs might be different
    train_dataset = VOCDataset('trainval', image_size = 512)
    val_dataset = VOCDataset('test', image_size=512)
    train_sampler = None


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True, 
        collate_fn= collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True, 
        collate_fn= collate_fn)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO (Q1.3): Create loggers for wandb.
    if USE_WANDB:
        wandb.init(project="vlr-hw1")
    # Ideally, use flags since wandb makes it harder to debug code.


    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)



# TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch):
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        input, target = data[0], data[1]
        input = input.to(device)
        target = target.to(device)

        # print(target)
        # print(target.shape)
        # TODO (Q1.1): Get output from model
        optimizer.zero_grad()
        imoutput = model(input)
        kernel_size = (list(imoutput.size())[2], list(imoutput.size())[3])
        # print(imoutput.shape)

        if (epoch == 0 or epoch == 14 or epoch == 29 ) and (i == 60 or i == 120) and USE_WANDB:
            # indices = [10, 11, 12, 14]
            # for index in indices:
            index = 6
            classes = data[5]
            resized_output = F.interpolate(imoutput, (512, 512))
            output = resized_output[index, classes[index][0],:,:]
            print("For:", i, "Label is:", classes[index])
            # normalized_output = torch.sigmoid(perspective)
            heatmap_image = output.cpu().detach().numpy()
            # cmap = plt.cm.jet
            # norm = plt.Normalize(vmin=normalized_output.min(), vmax=normalized_output.max())
            # heatmap_image = cmap(norm(normalized_output))
            plt.imsave('heatmap.png', heatmap_image, cmap = 'jet')
            # heatmap_image = heatmap_image.resize((512, 512))
            heatmap_image = Image.open('heatmap.png')
            img = tensor_to_PIL(input[index,:,:,:])
            plt.imshow(img)
            train_img = wandb.Image(img, caption= 'Image')
            train_heatmap = wandb.Image(heatmap_image, caption= 'Train/Heatmap')

            wandb.log({"Index: {},Epoch: {}, Class: {}".format(index,epoch, VOCDataset.get_class_name(index)): train_heatmap}) 
            wandb.log({ "Train/Index: {}, Train/Class: {}".format(index, VOCDataset.get_class_name(index)): train_img})

        imoutput = nn.MaxPool2d(kernel_size = kernel_size)(imoutput)
        imoutput = torch.squeeze(imoutput, -1)
        imoutput = torch.squeeze(imoutput, -1)

        # TODO (Q1.1): Perform any necessary operations on the output

        # TODO (Q1.1): Compute loss using ``criterion``
        loss = criterion(imoutput, target)

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # TODO (Q1.1): compute gradient and perform optimizer step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # TODO (Q1.3): Visualize/log things as mentioned in handout at appropriate intervals
        if USE_WANDB:
            wandb.log({'train/iteration': i, 'train/loss': loss})
            wandb.log({'train/iteration': i, 'train/mertic1': m1})
            wandb.log({'train/iteration': i, 'train/metric2': m2})
        # End of train()


def validate(val_loader, model, criterion, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data) in enumerate(val_loader):

        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        input, target = data[0], data[1]
        input = input.to(device)
        target = target.to(device)
        # TODO (Q1.1): Get output from model
        imoutput = model(input)
        kernel_size = (list(imoutput.size())[2], list(imoutput.size())[3])
        
        # if epoch == (args.epochs - 1) and USE_WANDB:
        if (i == 50 or i == 70 or i == 100) and (epoch == 29) and USE_WANDB:
            # indices = [10, 11, 12, 13, 14]
            # for index in indices:
            index = 6
            classes = data[5]
            resized_output = F.interpolate(imoutput, (512, 512))
            output = resized_output[index, classes[index][0],:,:]
            print("For:", i, "Label is:", classes[index][0])
            heatmap_image = output.cpu().detach().numpy()
            plt.imsave('heatmap.png', heatmap_image, cmap = 'jet')
            heatmap_image = Image.open('heatmap.png')
            img = tensor_to_PIL(input[index,:,:,:])
            plt.imshow(img)
            img = wandb.Image(img, caption= 'Image')
            heatmap = wandb.Image(heatmap_image, caption= 'Validate/Heatmap')

            wandb.log({"Index: {},Epoch: {}, Class: {}".format(index, epoch, VOCDataset.get_class_name(index)): heatmap}) 
            wandb.log({ "Val/Index: {}, Val/Class: {}".format(index, VOCDataset.get_class_name(index)): img})

        imoutput = nn.MaxPool2d(kernel_size = kernel_size)(imoutput)
        imoutput = torch.squeeze(imoutput, -1)
        imoutput = torch.squeeze(imoutput, -1)
        # print("This is output shape:", imoutput.shape)


        # TODO (Q1.1): Perform any necessary functions on the output

        # TODO (Q1.1): Compute loss using ``criterion``
        loss = criterion(imoutput, target)

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # TODO (Q1.3): Visualize things as mentioned in handout
    if USE_WANDB:
        wandb.log({'validate/epoch': epoch, 'validate/loss': loss})
        # wandb.log({'validate/iteration': i, 'validate/mertic1': m1})
        # wandb.log({'validate/iteration': i, 'validate/metric2': m2})

    if USE_WANDB and epoch%2==0:
        wandb.log({'validate/epoch': epoch, 'validate/avg_m1': avg_m1.avg})
        wandb.log({'validate/epoch': epoch, 'validate/avg_m2': avg_m2.avg})

        # TODO (Q1.3): Visualize at appropriate intervals




    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def metric1(output, target):
    # TODO (Q1.5): compute metric1
    sigmoid_output = torch.sigmoid(output)
    target = target.cpu().detach().numpy().astype('float32')
    processed_output = sigmoid_output.cpu().detach().numpy().astype('float32')
    
    average_precision = []
    for i in range(target.shape[1]):
        if target[:, i].any():
            average_precision.append(sklearn.metrics.average_precision_score(target[:, i], processed_output[:, i], average = None))
        # else:
        #      average_precision.append(0)
    mean_average_precision = np.mean(average_precision)


    return mean_average_precision


def metric2(output, target):
    # TODO (Q1.5): compute metric2
    sigmoid_output = torch.sigmoid(output).cpu().detach()
    # threshold = torch.tensor([0.5])
    # output_result = (sigmoid_output > threshold).float()*1
    target = target.cpu().detach().numpy().astype('float32')
    processed_output = sigmoid_output.numpy().astype('float32')

    zero_class = []
    for i in range(target.shape[-1]):
        if np.count_nonzero(target[:, i]) == 0:
            zero_class.append(i)
    
    # target = np.delete(target, zero_class, 1)
    # processed_output = np.delete(processed_output, zero_class, 1)
    processed_output = (processed_output >= 0.5)*1.0
    
    recall = sklearn.metrics.recall_score(target, processed_output, average='micro')


    return recall


if __name__ == '__main__':
    main()