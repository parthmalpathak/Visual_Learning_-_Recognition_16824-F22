from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import argparse
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl
import torchvision
from task_1 import USE_WANDB

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL, iou
from PIL import Image, ImageDraw, ImageFont
import cv2 as cv 


# hyper-parameters
# ------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--lr',
    default=0.0001,
    type=float
    # description='Learning rate'
)
parser.add_argument(
    '--lr-decay-steps',
    default=150000,
    type=int
    # description='Interval at which the lr is decayed'
)
parser.add_argument(
    '--lr-decay',
    default=0.1,
    type=float,
    # description='Decay rate of lr'
)
parser.add_argument(
    '--momentum',
    default=0.9,
    type=float
    # description='Momentum of optimizer'
)
parser.add_argument(
    '--weight-decay',
    default=0.0005,
    type=float
    # description='Weight decay'
)
parser.add_argument(
    '--epochs',
    default=5,
    type=int
    # description='Number of epochs'
)
parser.add_argument(
    '--val-interval',
    default=5000,
    type=int
    # description='Interval at which to perform validation'
)
parser.add_argument(
    '--disp-interval',
    default=10,
    type=int
    # description='Interval at which to perform visualization'
)
parser.add_argument(
    '--use-wandb',
    default=False,
    type=bool
    # description='Flag to enable visualization'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_WANDB = True
if USE_WANDB:
    wandb.init(project="vlr-hw1")
# ------------

# Set random seed
rand_seed = 1024
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

# Set output directory
output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def calculate_map():
    """
    Calculate the mAP for classification.
    """
    # TODO (Q2.3): Calculate mAP on test set.
    # Feel free to write necessary function parameters.

    pass



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



def test_model(model, val_loader=None, thresh=0.05):
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold
    """
    # losses = AverageMeter()
    with torch.no_grad():
        for iter, data in enumerate(val_loader):

            # one batch = data for one image
            image = data['image']
            target = data['label']
            wgt = data['wgt']
            rois = data['rois']*512
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']

            # TODO (Q2.3): perform forward pass, compute cls_probs
            image = image.to(device)
            target = target.to(device)
            rois = rois.to(device, dtype = torch.float)

            cls_probs = model(image, rois = rois, gt_vec = target)



            tp = 0
            fp = 0
            recall_den = 0
            nms_boxes = []
            nms_scores = []
            labels = []
            average_precision_list = np.zeros(20)
            # TODO (Q2.3): Iterate over each class (follow comments)
            for class_num in range(20):
                # get valid rois and cls_scores based on thresh
                cls_scores = cls_probs[:, class_num]
                shortlist = cls_scores > thresh
                selected_rois = rois[0, shortlist]
                selected_cls_scores = cls_scores[shortlist]
                boxes, scores = nms(selected_rois, selected_cls_scores, thresh)
                nms_boxes.extend(boxes)
                nms_scores.extend(scores)
                labels.append(VOCDataset.CLASS_NAMES[class_num])
                


                # # Calculation for AP
                # for i in range (len(scores)):
                #     maximum_iou = 0
                #     selected_rois = boxes[i]

                #     for j, gt_box in enumerate(gt_boxes): 
                #         if gt_class_list[j] == class_num: #Check for presence of gt_box in image
                #             recall_den += 1
                #             gt_roi = gt_box
                #             intersection_of_union = iou(selected_rois, gt_roi) #Compute IOU
                #             if intersection_of_union > maximum_iou:
                #                 maximum_iou = intersection_of_union #Substitute iou as maximum if computed value is greater
                #                 detected_gt = j
                #     if maximum_iou > 0.3: #Condition for True Positive and popping the GTs already taken care of.
                #         tp += 1
                #         gt_boxes.pop(detected_gt)
                #     else: 
                #         fp += 1

                # precision = tp / (tp + fp) #Formula for Precision
                # recall = tp / recall_den #Formula for recall
                # average_precision = precision * recall 
                # average_precision_list[class_num] = average_precision #Find Average Precision and store


                # detections = []
                # gt = []
                # for detection in  boxes:
                # use NMS to get boxes and scores

                

            # TODO (Q2.3): visualize bounding box predictions when required
            calculate_map()
            if iter%500 == 0 and USE_WANDB:
              image = image.cpu()
              image = tensor_to_PIL(image[0])

              font = ImageFont.load_default()
              draw = ImageDraw.Draw(image)  
              for box_i, box in enumerate(nms_boxes):
                box = box.cpu().detach().numpy().tolist()
                score = nms_scores[box_i].cpu().detach()
                label = labels[box_i]
                draw.rectangle(box, width = 3,outline='red')
                draw.text((box[0],box[3]), "iteration: {iter}, Class: {label}, Score: {score:.4f}".format(iter= iter, label = label, score = score), font=font, fill=(255, 255, 255, 255))
                
              wandb.log({"Image {}".format(iter) :wandb.Image(image)})


def train_model(model, train_loader=None, val_loader=None, optimizer=None, args=None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    # Initialize training variables
    losses = AverageMeter()
    train_loss = 0
    step_cnt = 0
    for epoch in range(args.epochs):
        for iter, data in enumerate(train_loader):

            # TODO (Q2.2): get one batch and perform forward pass
            # one batch = data for one image
            image = data['image']
            target = data['label']
            wgt = data['wgt']
            rois = data['rois']*512
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']

            # TODO (Q2.2): perform forward pass
            # take care that proposal values should be in pixels
            # Convert inputs to cuda if training on GPU
            image = image.to(device)
            target = target.to(device)
            rois = rois.to(device, dtype = torch.float)

            cls_probs = model(image, rois = rois, gt_vec = target)
            



            # backward pass and update
            loss = model.loss
            train_loss += loss.item()
            step_cnt += 1
            losses.update(loss.item(), n = image.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO (Q2.2): evaluate the model every N iterations (N defined in handout)
            # Add wandb logging wherever necessary
            if iter % args.val_interval == 0 and iter != 0:
                model.eval()
                ap = test_model(model, val_loader)
                print("AP ", ap)
                model.train()

            # TODO (Q2.4): Perform all visualizations here
            # The intervals for different things are defined in the handout
            if iter % 500 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                 'Loss: {loss:.4f}'.format(
                      epoch,
                      iter,
                      len(train_loader),
                      loss=losses.avg))

            if USE_WANDB:
                wandb.log({'train/iteration': iter, 'train/loss': loss})
    # TODO (Q2.4): Plot class-wise APs


def main():
    """
    Creates dataloaders, network, and calls the trainer
    """
    args = parser.parse_args()
    # TODO (Q2.2): Load datasets and create dataloaders
    # Initialize wandb logger
    train_dataset = VOCDataset('trainval', 512, top_n = 300)
    val_dataset = VOCDataset('test', 512, top_n = 300)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,   # batchsize is one for this implementation
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    # Create network and initialize
    net = WSDDN(classes=train_dataset.CLASS_NAMES)
    print(net)

    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net,
        open('pretrained_alexnet.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()

    for name, param in pret_net.items():
        print(name)
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            print('Copied {}'.format(name))
        except:
            print('Did not find {}'.format(name))
            continue

    # Move model to GPU and set train mode
    net.load_state_dict(own_state)
    net.cuda()
    net.train()

    # TODO (Q2.2): Freeze AlexNet layers since we are loading a pretrained model
    for param in net.features:
        param.requires_grad_ = False

    # TODO (Q2.2): Create optimizer only for network parameters that are trainable
    optimizer = torch.optim.SGD(net.parameters(), lr= args.lr, momentum = args.momentum)

    # Training
    train_model(net, train_loader = train_loader, val_loader = val_loader, optimizer = optimizer, args= args)

if __name__ == '__main__':
    main()

