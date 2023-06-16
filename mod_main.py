# Code that takes args from user, creates model, dataloaders 
# trains model, validates model, and saves model

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from time import gmtime, strftime
from tqdm import tqdm
import random
import json
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from model import Sal_seq, SalBert, CaptionModel, VisualCaptionModel
from data import read_dataset, ASDDataset, CaptionDataset

parser = argparse.ArgumentParser(description='Autism screening based on eye-tracking data')
parser.add_argument('--img_dir', type=str, default='TrainingDataset\TrainingData\Images', help='Directory to images')
parser.add_argument('--anno_dir', type=str, default='TrainingDataset\TrainingData',
                    help='Directory to annotation files')
parser.add_argument('--backend', type=str, default='resnet', help='Backend for visual encoder')
parser.add_argument('--lr', type=float, default=1e-5, help='specify learning rate')
parser.add_argument('--checkpoint_path', type=str, default='Checkpoints', help='Directory for saving checkpoints')
parser.add_argument('--epoch', type=int, default=20, help='Specify maximum number of epoch')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
parser.add_argument('--max_len', type=int, default=14, help='Maximum number of fixations for an image')
parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size for RNN/BERT')
parser.add_argument('--clip', type=float, default=0, help='Gradient clipping')
parser.add_argument('--img_height', type=int, default=600, help='Image Height')
parser.add_argument('--img_width', type=int, default=800, help='Image Width')
parser.add_argument('--t_i', type=int, default=1, help='Start index of training data')
parser.add_argument('--t_f', type=int, default=241, help='End index of training data')
parser.add_argument('--v_i', type=int, default=241, help='Start index of validation data')
parser.add_argument('--v_f', type=int, default=301, help='End index of validation data')

# Time Relted Arguments
parser.add_argument('--mask', type=bool, default=False, help='Time Masking')
parser.add_argument('--joint', type=bool, default=False, help='Joint Time Embedding')

# Model Related Arguments
parser.add_argument('--model', type=str, default='VisualCaption', help='Model to use')

args = parser.parse_args()

transform = transforms.Compose([transforms.Resize((args.img_height, args.img_width)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_crop = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch):
    "adatively adjust lr based on epoch"
    if epoch <= 0:
        lr = args.lr
    else:
        lr = args.lr * (0.5 ** (float(epoch) / 2))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, trainloader, optimizer, criterion, iteration, arg):
    avg_loss = 0
    preds = []
    targets = []
    model.train()
    for i, data in enumerate(tqdm(trainloader)):
        optimizer.zero_grad()
        if arg == 'Visual':
            img, label, fixation, duration = data
            if len(img) == 1:  # Skip if batch size is 1
                continue
            duration = duration.type(torch.FloatTensor)
            if torch.cuda.is_available():  # Move to GPU if available
                img = img.cuda()
                label = label.cuda()
                fixation = fixation.cuda()
                duration = duration.cuda()
            pred = model(img, fixation, duration)  # Forward pass

        elif arg == 'CaptionOnly':
            fix_caps, label = data
            if torch.cuda.is_available():  # Move to GPU if available
                label = label.cuda()
                for k, v in fix_caps.items():
                    fix_caps[k] = v.cuda()
            pred = model(fix_caps)  # Forward pass

        else:  # VisualCaptionModel
            fix_crops, fix_caps, label = data
            if torch.cuda.is_available():  # Move to GPU if available
                label = label.cuda()
                for k, v in fix_caps.items():
                    fix_caps[k] = v.cuda()
                fix_crops = [c.cuda() for c in fix_crops]
            pred = model(fix_crops, fix_caps)  # Forward pass

        # Compute loss
        loss = criterion(pred, label)

        # Backward pass
        loss.backward()

        if args.clip > 0:  # Gradient clipping
            clip_gradient(optimizer, args.clip)

        # Update weights
        optimizer.step()

        # Log loss on wandb
        avg_loss = (avg_loss * np.maximum(0, i) + loss.data.cpu().numpy()) / (i + 1)
        if i % 25 == 0:
            wandb.log({'bce loss': avg_loss}, step=iteration)

        iteration += 1

        # Store predictions and targets for computing metrics
        preds.append(pred.detach().cpu().numpy())
        targets.append(label.detach().cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Compute metrics
    acc = accuracy_score(targets, preds > 0.5)
    auc = roc_auc_score(targets, preds)
    precision = precision_score(targets, preds > 0.5)
    recall = recall_score(targets, preds > 0.5)
    f1 = f1_score(targets, preds > 0.5)

    # Print metrics in one line
    print(
        "Train: Acc: {:.4f}, AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Loss: {:.4f}".format(acc, auc,
                                                                                                              precision,
                                                                                                              recall,
                                                                                                              f1,
                                                                                                              avg_loss))

    # Log metrics on wandb
    wandb.log({'train acc': acc, 'train auc': auc, 'train precision': precision, 'train recall': recall, 'train f1': f1,
               'train loss': avg_loss})

    return iteration


# Validation Function
def validate(model, valloader, criterion, arg):
    preds = []
    targets = []
    model.eval()
    avg_loss = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(valloader)):
            if arg == 'Visual':
                img, label, fixation, duration = data
                if len(img) == 1:  # Skip if batch size is 1
                    continue
                duration = duration.type(torch.FloatTensor)
                if torch.cuda.is_available():  # Move to GPU if available
                    img = img.cuda()
                    label = label.cuda()
                    fixation = fixation.cuda()
                    duration = duration.cuda()
                pred = model(img, fixation, duration)  # Forward pass

            elif arg == 'CaptionOnly':
                fix_caps, label = data
                if torch.cuda.is_available():  # Move to GPU if available
                    label = label.cuda()
                    for k, v in fix_caps.items():
                        fix_caps[k] = v.cuda()

                pred = model(fix_caps)  # Forward pass

            else:  # VisualCaptionModel
                fix_crops, fix_caps, label = data
                if torch.cuda.is_available():  # Move to GPU if available
                    label = label.cuda()
                    for k, v in fix_caps.items():
                        fix_caps[k] = v.cuda()
                    fix_crops = [c.cuda() for c in fix_crops]

                pred = model(fix_crops, fix_caps)  # Forward pass

            loss = criterion(pred, label)
            avg_loss = (avg_loss * np.maximum(0, i) + loss.data.cpu().numpy()) / (i + 1)

            preds.append(pred.detach().cpu().numpy())
            targets.append(label.detach().cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Compute metrics
    acc = accuracy_score(targets, preds > 0.5)
    auc = roc_auc_score(targets, preds)
    precision = precision_score(targets, preds > 0.5)
    recall = recall_score(targets, preds > 0.5)
    f1 = f1_score(targets, preds > 0.5)

    # Print metrics in one line
    print("Val: Acc: {:.4f}, AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Loss: {:.4f}".format(acc, auc,
                                                                                                              precision,
                                                                                                              recall,
                                                                                                              f1,
                                                                                                              avg_loss))

    # Log metrics on wandb
    wandb.log({'val acc': acc, 'val auc': auc, 'val precision': precision, 'val recall': recall, 'val f1': f1,
               'val loss': avg_loss})

    return acc


def main(name_of_project, name_of_run, train_idx, val_idx, mode):
    wandb.init(project=name_of_project, name=name_of_run)
    args.model = mode
    # Load Data
    print("#" * 8)
    print("Loading data")
    dat = {}
    with open('caption_annotations.json', 'r') as f:
        dat = json.load(f)
    dat = {int(k): dat[k] for k in dat.keys()}
    train_capAnno = {k: dat[k] for k in train_idx}
    val_capAnno = {k: dat[k] for k in val_idx}

    train_fixAnno = read_dataset(args.anno_dir, train_idx)
    val_fixAnno = read_dataset(args.anno_dir, val_idx)

    print("Length of training data: ", len(train_fixAnno))
    print("Length of validation data: ", len(val_fixAnno))

    # Data Loaders
    print("#" * 8)
    print("Creating data loaders")
    if args.model == 'CaptionOnly':
        train_set = CaptionDataset(train_capAnno, args.max_len)
        val_set = CaptionDataset(val_capAnno, args.max_len)
    elif args.model == 'VisualCaption':
        train_set = CaptionDataset(train_capAnno, args.max_len, True, train_fixAnno, transform=transform_crop)
        val_set = CaptionDataset(val_capAnno, args.max_len, True, val_fixAnno, transform=transform_crop)
    else:
        train_set = ASDDataset(args.img_dir, train_fixAnno, args.max_len, args.img_height, args.img_width, transform)
        val_set = ASDDataset(args.img_dir, val_fixAnno, args.max_len, args.img_height, args.img_width, transform)

    print("Length of training set: ", len(train_set))
    print("Length of validation set: ", len(val_set))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    # Model
    print("Creating model")
    if args.model == 'Sal_seq':
        model = Sal_seq(backend=args.backend, seq_len=args.max_len,
                        hidden_size=args.hidden_size,
                        mask=args.mask, joint=args.joint)
        print("Created Sal_seq model")
    elif args.model == 'SalBert':
        model = SalBert(backend=args.backend, seq_len=args.max_len)
        print("Created SalBert model")
    elif args.model == 'CaptionOnly':
        model = CaptionModel(seq_len=args.max_len, hidden_size=args.hidden_size)
        print("Created CaptionOnly Model")
    elif args.model == 'VisualCaption':
        model = VisualCaptionModel(seq_len=args.max_len, hidden_size=args.hidden_size)
        print("Create VisualCaption Model")
    else:
        print("Model not recognized")
        return

    total_params = sum([p.numel() for p in filter(lambda p: p.requires_grad, model.parameters())])
    print(f"Total trainable parameters in model: {total_params:,}")

    if args.mask or args.joint:
        print("Mask: ", args.mask)
        print("Joint: ", args.joint)
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model on current device: ", torch.cuda.current_device())
    else:
        print('CUDA is not available. Model on CPU')

    # Optimizer
    print("Creating optimizer")
    if mode in ["SalBert", "CaptionOnly"]:
        args.lr = 1e-5
    else:
        args.lr = 1e-4
    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters())}], lr=args.lr,
                                 weight_decay=1e-5)

    # Loss
    print("Creating loss function")
    criterion = nn.BCELoss()

    # Training Loop
    print("#" * 8)
    print("Training loop")
    iteration = 0
    best_acc = 0
    if mode in ["Sal_seq", "SalBert"]:
        arg = 'Visual'
    elif mode == "CaptionOnly":
        arg = 'CaptionOnly'
    else:
        arg = "VisualCaption"

    os.makedirs(f"{args.checkpoint_path}\{name_of_project}\{name_of_run}", exist_ok=True)

    for epoch in range(args.epoch):
        print("\nEpoch: ", epoch + 1)
        iteration = train(model, trainloader, optimizer, criterion, iteration, arg)
        acc = validate(model, valloader, criterion, arg)
        if acc > best_acc:
            best_acc = acc
            model_name = f"{name_of_project}-{name_of_run}-bestacc-{acc:.2f}.pth"
            model_path = os.path.join(f"{args.checkpoint_path}\{name_of_project}\{name_of_run}", model_name)
            torch.save(model.state_dict(), model_path)
            print("Saved " + model_name)

        if mode == "Sal_Seq":
            adjust_lr(optimizer, epoch)

        model_name = f"{name_of_project}-{name_of_run}-epoch-{epoch + 1}.pth"
        model_path = os.path.join(f"{args.checkpoint_path}\{name_of_project}\{name_of_run}", model_name)
        torch.save(model.state_dict(), model_path)
        print("Saved " + model_name)
        
    wandb.finish()


if __name__ == "__main__":
    for mode in ["Sal_seq", "SalBert"]:
        name_of_project = "asd_runs"
        for i in range(3):
            name_of_run = mode + f"_run_{i}"
            all_idx = list(range(1, 301))
            val_idx = random.sample(range(1,301), 60)
            train_idx = set(val_idx) ^ set(all_idx)
            main(name_of_project, name_of_run, train_idx, val_idx, mode)
