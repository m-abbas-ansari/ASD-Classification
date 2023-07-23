# Code that takes args from user, creates model, dataloaders 
# trains model, validates model, and saves model

import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from time import gmtime, strftime
from tqdm import tqdm
import json
import math
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from model import LARS, HATFormer, HATLastFormer, HATFormerTensor, Sal_seq_pre, SSL
from data import read_dataset, FixDataset, FixTensorDatasetCPU, FixViewDataset

from fine_tune import main as finetune

IMG_HEIGHT = 320
IMG_WIDTH = 512
BASE_LR = 0.2
LR_WEIGHTs = 0.2
LR_BIASES = 0.0048
WEIGHT_DECAY = 1e-6
#ANNO_DIR = "Tensor-Dataset"
ANNO_DIR = "../../Data/FIXATION_DATASET"
SEED = 42
VAL_RATIO = 0.2
NUM_FIX = 20
BATCH_SIZE = 64
EPOCHS = 20

# Set Seeds for reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


transform = transforms.Compose([transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



def adjust_learning_rate(optimizer, loader, step):
    max_steps = EPOCHS * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = BASE_LR * BATCH_SIZE / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * LR_WEIGHTs
    optimizer.param_groups[1]['lr'] = lr * LR_BIASES

def get_last_fix(x, lengths):
    batch_size = x.size(0)
    seq_length = x.size(1)
    mask = x.data.new().resize_as_(x.data).fill_(0)
    for i in range(batch_size):
        mask[i][lengths[i] - 1].fill_(1)
    mask = Variable(mask)
    x = x.mul(mask)
    x = x.sum(1).view(batch_size, x.size(2))
    
    return x

def train(model, trainloader, optimizer, epoch): #, loss_fn_token, loss_fn_y, loss_fn_x):
    # token_losses = 0
    # reg_losses = 0
    losses = 0
    model.train()
    with tqdm(trainloader, unit='batch') as tepoch:
        minibatch = 0
        for step, data in enumerate(tepoch, start=epoch * len(trainloader)):
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, trainloader, step)
            # #fixs, pad_mask, per_tokens, fov_tokens = data
            # imgs, fixs, pad_mask = data
            # if len(imgs) == 1:
            #     continue
            
            # if torch.cuda.is_available():  # Move to GPU if available
            #     fixs = fixs.cuda()
            #     pad_mask = pad_mask.cuda()
            #     imgs = imgs.cuda()
            
            view1, view2 = data
            img1, fix1, pad1 = view1
            if len(img1) == 1:
                continue
            img2, fix2, pad2 = view2
            img1, fix1, pad1 = img1.cuda(), fix1.cuda(), pad1.cuda()
            img2, fix2, pad2 = img2.cuda(), fix2.cuda(), pad2.cuda()
            loss = model((img1, fix1, pad1), (img2, fix2, pad2))
            
            # #out_token, out_y, out_x = model(fixs, pad_mask, per_tokens, fov_tokens)  # Forward pass
            # out_token, out_y, out_x = model(imgs, fixs, pad_mask)  # Forward pass
            # out_token = out_token.transpose(0,1)
            # out_y = out_y.transpose(0,1)
            # out_x = out_x.transpose(0,1)
            
            # tgt_out = fixs
            # batch_tgt_padding_mask = pad_mask[:, -20:]
            # token_gt = batch_tgt_padding_mask.long()
            # fixation_mask = torch.logical_not(batch_tgt_padding_mask).float()
            # #predict padding or valid fixation
            # token_loss = loss_fn_token(out_token.permute(1,2,0), token_gt)
            # out_y = out_y.squeeze(-1).permute(1,0) * fixation_mask
            # out_x = out_x.squeeze(-1).permute(1,0) * fixation_mask
            # #calculate regression L1 losses for only valid ground truth fixations
            # reg_loss = (loss_fn_y(out_y.float(), tgt_out[:, :, 0] * fixation_mask).sum(-1)/fixation_mask.sum(-1) + loss_fn_x(out_x.float(), tgt_out[:, :, 1]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()
            # loss = token_loss + reg_loss
            loss.backward()
            # token_losses += token_loss.item()
            # reg_losses += reg_loss.item()
            losses += loss
            # Update weights
            optimizer.step()

            # Log loss on wandb
            minibatch += 1.
            # tepoch.set_postfix(token_loss=token_losses/minibatch, reg_loss=reg_losses/minibatch, )
            tepoch.set_postfix(loss=losses/minibatch )

            wandb.log({'loss iter': losses/minibatch})

    # Compute metrics

    # Print metrics in one line
    # avg_loss = (token_losses + reg_losses) / len(trainloader)
    avg_loss = losses/len(trainloader)
    print(
        "Train: Loss: {:.4f}".format(avg_loss))
    # Log metrics on wandb
    # wandb.log({'train token loss': token_losses/ len(trainloader), 'train reg loss': reg_losses/ len(trainloader)})
    wandb.log({'train loss': losses/ len(trainloader)})



# Validation Function
def validate(model, valloader): #, loss_fn_token, loss_fn_y, loss_fn_x):
    # token_losses = 0
    # reg_losses = 0
    losses = 0
    model.eval()

    with torch.no_grad():
        with tqdm(valloader, unit='batch') as vepoch:
            minibatch = 0
            for data in vepoch:                
                #fixs, pad_mask, per_tokens, fov_tokens = data
                # imgs, fixs, pad_mask = data
                # if len(imgs) == 1:
                #     continue
                
                # if torch.cuda.is_available():  # Move to GPU if available
                #     fixs = fixs.cuda()
                #     pad_mask = pad_mask.cuda()
                #     imgs = imgs.cuda()
                    
                view1, view2 = data
                img1, fix1, pad1 = view1
                if len(img1) == 1:
                    continue
                img2, fix2, pad2 = view2
                img1, fix1, pad1 = img1.cuda(), fix1.cuda(), pad1.cuda()
                img2, fix2, pad2 = img2.cuda(), fix2.cuda(), pad2.cuda()
                loss = model((img1, fix1, pad1), (img2, fix2, pad2))
                
                # #out_token, out_y, out_x = model(fixs, pad_mask, per_tokens, fov_tokens)  # Forward pass
                # out_token, out_y, out_x = model(imgs, fixs, pad_mask)  # Forward pass
                # out_token = out_token.transpose(0,1)
                # out_y = out_y.transpose(0,1)
                # out_x = out_x.transpose(0,1)
                
                # tgt_out = fixs
                # batch_tgt_padding_mask = pad_mask[:, -20:]
                # token_gt = batch_tgt_padding_mask.long()
                # fixation_mask = torch.logical_not(batch_tgt_padding_mask).float()
                # #predict padding or valid fixation
                # token_loss = loss_fn_token(out_token.permute(1,2,0), token_gt)
                # out_y = out_y.squeeze(-1).permute(1,0) * fixation_mask
                # out_x = out_x.squeeze(-1).permute(1,0) * fixation_mask
                # #calculate regression L1 losses for only valid ground truth fixations
                # reg_loss = (loss_fn_y(out_y.float(), tgt_out[:, :, 0] * fixation_mask).sum(-1)/fixation_mask.sum(-1) + loss_fn_x(out_x.float(), tgt_out[:, :, 1]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()
                # loss = token_loss + reg_loss
                # token_losses += token_loss.item()
                # reg_losses += reg_loss.item()
                losses += loss
                minibatch += 1.
                vepoch.set_postfix(loss=losses/minibatch)

    # Print metrics in one line
    # avg_loss = (token_losses + reg_losses) / len(valloader)
    avg_loss = losses/len(valloader)
    print("Val: Loss: {:.4f}".format(avg_loss))

    # Log metrics on wandb
    wandb.log({'val  loss': avg_loss})
    
    return avg_loss


def main_train(project, method, run):
    print("#" * 8)
    print("Wandb setup")
    name_of_project = project
    name_of_run = "{}-{}".format(method, run)
    print("Training\nProject: {} Run: {}".format(name_of_project, name_of_run))
    # name_of_project = "ssl"
    # name_of_run = "fast-af-boi"
    wandb.init(project=name_of_project, name=name_of_run)

    # Load Data
    print("#" * 8)
    print("Loading data")
    #dataset = FixTensorDatasetCPU(ANNO_DIR)
    datasets = ['SIENA12', 'OSIE', 'FIWI', 'VIU']
    anno, _ = read_dataset(ANNO_DIR, datasets, val_ratio=0.0)
    dataset = FixViewDataset(anno, augmentations=[run], transform=transform)
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = torch.utils.data.random_split(dataset, 
                                                       [1 - VAL_RATIO, VAL_RATIO], 
                                                       generator=generator)
  

    print("Length of training data: ", len(train_set))
    print("Length of validation data: ", len(val_set))
    
    # Data Loaders
    print("#" * 8)
    print("Creating data loaders")
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    print("trainloader: ", trainloader)
    print("valloader: ", valloader)
    
    # Model
    print("Creating model")
    #model = HATFormerTensor()
    model = SSL(method=method, visual_backend="resnet18", hidden_dim=256, batch_size=BATCH_SIZE)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    total_params = sum([p.numel() for p in filter(lambda p: p.requires_grad, model.parameters())])
    print(f"Total trainable parameters in model: {total_params:,}")

    if torch.cuda.is_available():
        model = model.cuda()
        print("Model on current device: ", torch.cuda.current_device())
    else:
        print('CUDA is not available. Model on CPU')

    # Optimizer
    print("Creating optimizer")
    optimizer = LARS(parameters, lr=0, weight_decay=WEIGHT_DECAY,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)
    # optimizer = torch.optim.AdamW([{'params': filter(lambda p: p.requires_grad, model.parameters())}], lr=LR,
    #                              weight_decay=1e-5)

    # Loss
    # print("Creating loss function")
    # loss_fn_token = torch.nn.NLLLoss()
    # loss_fn_y = nn.L1Loss(reduction='none')
    # loss_fn_x = nn.L1Loss(reduction='none')

    # Training Loop
    print("#" * 8)
    print("Training loop")
    os.makedirs(f"Checkpoints/{name_of_project}/{name_of_run}", exist_ok=True)
    #os.makedirs(f"{args.checkpoint_path}\{name_of_project}\{name_of_run}", exist_ok=True)
    for epoch in range(EPOCHS):
        print("\nEpoch: ", epoch + 1)
        train(model, trainloader, optimizer, epoch) #loss_fn_token, loss_fn_y, loss_fn_x)
        val_loss = validate(model, valloader) #loss_fn_token, loss_fn_y, loss_fn_x)
        torch.save(model, f"Checkpoints/{name_of_project}/{name_of_run}/epoch-{epoch}.pt")
    wandb.finish()

if __name__ == "__main__":
    
    name_of_runs = [
        'noise_addition',
        'horizontal_flip',
        'segment_deletion',
        'scanpath_reversal',
        'global_rotation_w_par_fix',
        'global_rotation_only_img',
        'global_rotation_w_fix'
    ]
    # finetune("asd-ssl", "horizontal_flip")
    for run in name_of_runs:
        main_train("ssl", "vic-reg", run)
        finetune("asd-ssl", "vic-reg", run)