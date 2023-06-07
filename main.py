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
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from model import Sal_seq
from data import read_dataset, ASDDataset

parser = argparse.ArgumentParser(description='Autism screening based on eye-tracking data')
parser.add_argument('--img_dir', type=str, default='../../Datasets/Saliency4ASD/TrainingData/Images', help='Directory to images')
parser.add_argument('--anno_dir', type=str, default='../../Datasets/Saliency4ASD/TrainingData', help='Directory to annotation files')
parser.add_argument('--backend', type=str, default='resnet', help='Backend for visual encoder')
parser.add_argument('--lr',type=float,default=1e-4,help='specify learning rate')
parser.add_argument('--checkpoint_path',type=str,default='Checkpoints',help='Directory for saving checkpoints')
parser.add_argument('--epoch',type=int,default=10,help='Specify maximum number of epoch')
parser.add_argument('--batch_size',type=int,default=12,help='Batch size')
parser.add_argument('--max_len',type=int,default=14,help='Maximum number of fixations for an image')
parser.add_argument('--hidden_size',type=int,default=512,help='Hidden size for RNN')
parser.add_argument('--clip',type=float,default=10,help='Gradient clipping')
parser.add_argument('--img_height',type=int,default=600,help='Image Height')
parser.add_argument('--img_width',type=int,default=800,help='Image Width')
parser.add_argument('--t_i', type=int, default=1, help='Start index of training data')
parser.add_argument('--t_f', type=int, default=241, help='End index of training data')
parser.add_argument('--v_i', type=int, default=241, help='Start index of validation data')
parser.add_argument('--v_f', type=int, default=301, help='End index of validation data')

args = parser.parse_args()

transform = transforms.Compose([transforms.Resize((args.img_height,args.img_width)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, epoch):
    "adatively adjust lr based on epoch"
    if epoch <= 0 :
        lr = args.lr
    else :
        lr = args.lr * (0.5 ** (float(epoch) / 2))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training Function 
def train(model, trainloader, optimizer, criterion, iteration):
    avg_loss = 0
    preds = []
    targets = []
    model.train()
    for i, (img, label, fixation, duration) in enumerate(tqdm(trainloader)):
        if len(img) == 1: # Skip if batch size is 1
            continue
        if torch.cuda.is_available(): # Move to GPU if available
            img = img.cuda()
            label = label.cuda()
            fixation = fixation.cuda()
            duration = duration.cuda()
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(img, fixation)
        
        # Compute loss
        loss = criterion(pred, label)
        
        # Backward pass
        loss.backward()
        
        if args.clip > 0: # Gradient clipping
            clip_gradient(optimizer, args.clip)
            
        # Update weights
        optimizer.step()
        
        # Log loss on wandb
        avg_loss = (avg_loss*np.maximum(0,i) + loss.data.cpu().numpy())/(i+1)
        if i%25 == 0:
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
    print("Train: Acc: {:.4f}, AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(acc, auc, precision, recall, f1))
    
    return iteration

# Validation Function
def validate(model, valloader, criterion):
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for img, label, fixation, duration in tqdm(valloader):
            if len(img) == 1: # Skip if batch size is 1
                continue
            if torch.cuda.is_available(): # Move to GPU if available
                img = img.cuda()
                label = label.cuda()
                fixation = fixation.cuda()
                duration = duration.cuda()

            pred = model(img, fixation)

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
    print("Val: Acc: {:.4f}, AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(acc, auc, precision, recall, f1))
    
    # Log metrics on wandb
    wandb.log({'val acc': acc, 'val auc': auc, 'val precision': precision, 'val recall': recall, 'val f1': f1})
    
    return auc
        
def main():
    print("#"*8)
    print("Wandb setup")
    name_of_run=str(input("Name of run: "))
    wandb.init(project="asd-lstm", name=name_of_run)
    
    # Load Data
    print("#"*8)
    print("Loading data")
    train_anno = read_dataset(args.anno_dir, args.t_i, args.t_f)
    val_anno = read_dataset(args.anno_dir, args.v_i, args.v_f)
    print("Length of training data: ", len(train_anno))
    print("Length of validation data: ", len(val_anno))
    
    # Data Loaders
    print("#"*8)
    print("Creating data loaders")
    train_set = ASDDataset(args.img_dir,train_anno,args.max_len,args.img_height,args.img_width,transform)
    val_set = ASDDataset(args.img_dir,val_anno,args.max_len,args.img_height,args.img_width,transform)
    print("Length of training set: ", len(train_set))
    print("Length of validation set: ", len(val_set))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    # Model
    print("Creating model")
    model = Sal_seq(backend=args.backend,seq_len=args.max_len,hidden_size=args.hidden_size)
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model on current device: ",torch.cuda.current_device())
    else:
        print('CUDA is not available. Model on CPU')
    
    # Optimizer
    print("Creating optimizer")
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=1e-5)
    
    # Loss
    print("Creating loss function")
    criterion = nn.BCELoss()
    
    # Training Loop
    print("#"*8)
    print("Training loop")
    iteration = 0
    best_auc = 0
    for epoch in range(args.epoch):
        print("\nEpoch: ", epoch+1)
        iteration = train(model, trainloader, optimizer, criterion, iteration)
        auc = validate(model, valloader, criterion)
        if auc > best_auc:
            best_auc = auc
            model_name = f"lstm-base-auc-{auc:.2f} " + strftime("%Y-%m-%d_%H:%M:%S", gmtime())+".pth"
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, model_name))
            print("Saved " + model_name)
        adjust_lr(optimizer, epoch)
        
main()