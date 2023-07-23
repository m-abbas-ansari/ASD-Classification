import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from time import gmtime, strftime
from tqdm import tqdm
import json
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from model import HATClassifier, Sal_seq
from data import read_ASD_dataset, ASDHATDataset

IMG_HEIGHT = 320
IMG_WIDTH = 512
HIDDEN_DIM = 256
LR = 1e-4
ANNO_DIR = "../../Data/Saliency4ASD/TrainingData"
IMG_DIR = "../../Data/Saliency4ASD/TrainingData/Images"
SEED = 42
VAL_RATIO = 0.2
NUM_FIX = 20
BATCH_SIZE = 32
EPOCHS = 20
SLOW_LR = 1e-5
FAST_LR = 1e-3

# Set Seeds for reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

transform = transforms.Compose([transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# decoder_p = nn.Linear(HIDDEN_DIM * 4, 2).to("cuda")
softmax = nn.LogSoftmax(dim=-1).to("cuda")

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch):
    "adaptively adjust lr based on epoch"
    if epoch <= 0:
        lr = LR
    else:
        lr = LR * (0.5 ** (float(epoch) / 2))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

def train(model, trainloader, SlowOpt, FastOpt, loss_fn_token, decoder):
    token_losses = 0
    preds = []
    targets = []
    
    model.train()
    decoder.train()
    with tqdm(trainloader, unit='batch') as tepoch:
        minibatch = 0
        for data in tepoch:
            SlowOpt.zero_grad()
            FastOpt.zero_grad()
            
            imgs, fixs, _, pad_mask, labels = data

            if torch.cuda.is_available():  # Move to GPU if available
                fixs = fixs.cuda()
                pad_mask = pad_mask.cuda()
                labels = labels.cuda().squeeze(1)
                imgs = imgs.cuda()

            # Forward pass               
            out = model.backbone(imgs, fixs, pad_mask)
            #out  = model.projector(out) 
            out_token = softmax(decoder(out)) 

            loss = loss_fn_token(out_token, labels)
          
            loss.backward()
            token_losses += loss.item()

            # Update weights
            SlowOpt.step()
            FastOpt.step()
            
            preds.append(torch.argmax(out_token, axis=-1).detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

          
            # Log loss on wandb
            minibatch += 1.
            tepoch.set_postfix(token_loss=token_losses/minibatch)
            wandb.log({'token loss iter': token_losses/minibatch})

    # Compute metrics
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
          
    avg_loss = (token_losses) / len(trainloader)
    acc = accuracy_score(targets, preds)
    auc = roc_auc_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
          
    # Print metrics in one line         
    print(
        "Train: Acc: {:.4f}, AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Loss: {:.4f}".format(acc, auc,
                                                                                                              precision,
                                                                                                              recall,
                                                                                                              f1,
                                                                                                              avg_loss))
    
    wandb.log({'train acc': acc, 'train auc': auc, 'train precision': precision, 'train recall': recall, 'train f1': f1,
               'train loss': avg_loss})


# Validation Function
def validate(model, valloader, loss_fn_token, decoder):
    token_losses = 0
    preds = []
    targets = []
    
    model.eval()
    decoder.eval()
    with torch.no_grad():
        with tqdm(valloader, unit='batch') as vepoch:
            minibatch = 0
            for data in vepoch:
        
                imgs, fixs, _, pad_mask, labels = data

                if torch.cuda.is_available():  # Move to GPU if available
                    fixs = fixs.cuda()
                    pad_mask = pad_mask.cuda()
                    labels = labels.cuda().squeeze(1)
                    imgs = imgs.cuda()

                # Forward pass               
                out = model.backbone(imgs, fixs, pad_mask)
                #out  = model.projector(out) 
                out_token = softmax(decoder(out))  

                loss = loss_fn_token(out_token, labels)
            
                token_losses += loss.item()

                preds.append(torch.argmax(out_token, axis=-1).detach().cpu().numpy())
                targets.append(labels.detach().cpu().numpy())

            
                # Log loss on wandb
                minibatch += 1.
                vepoch.set_postfix(token_loss=token_losses/minibatch)

    # Compute metrics
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
          
    avg_loss = (token_losses) / len(valloader)
    acc = accuracy_score(targets, preds)
    auc = roc_auc_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
          
    # Print metrics in one line         
    print(
        "val: Acc: {:.4f}, AUC: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Loss: {:.4f}".format(acc, auc,
                                                                                                              precision,
                                                                                                              recall,
                                                                                                              f1,
                                                                                                              avg_loss))
    
    wandb.log({'val acc': acc, 'val auc': auc, 'val precision': precision, 'val recall': recall, 'val f1': f1,
               'val loss': avg_loss})
    
    return avg_loss


def main(project, method, run):
    print("#" * 8)
    print("Wandb setup")
    name_of_project = project
    name_of_run = "{}-{}".format(method, run)
    print("Finetuning\nProject: {} Run: {}".format(name_of_project, name_of_run))
    # name_of_project = "ssl"
    # name_of_run = "fast-af-boi"
    wandb.init(project=name_of_project, name=name_of_run)

    # Load Data
    print("#" * 8)
    print("Loading data")
    anno = read_ASD_dataset(ANNO_DIR)
    dataset = ASDHATDataset(IMG_DIR, anno, transform=transform)
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


    # Model
    print("Creating model")
    model = torch.load(f"Checkpoints/ssl/{name_of_run}/epoch-19.pt")
    # model = BarlowTwins(visual_backend="resnet18", hidden_dim=256, batch_size=BATCH_SIZE)
    total_params = sum([p.numel() for p in filter(lambda p: p.requires_grad, model.parameters())])
    print(f"Total trainable parameters in model: {total_params:,}")

    if torch.cuda.is_available():
        model = model.cuda()
        print("Model on current device: ", torch.cuda.current_device())
    else:
        print('CUDA is not available. Model on CPU')

    # decoder = nn.Linear(HIDDEN_DIM, 2).to("cuda")
    decoder = nn.Sequential(*[nn.Linear(HIDDEN_DIM, HIDDEN_DIM), 
                            nn.ReLU(inplace=True),
                            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                            nn.ReLU(inplace=True),
                            nn.Linear(HIDDEN_DIM, 2)]).to("cuda")
    
    # Optimizer
    print("Creating optimizer")
    hat_params = list(model.parameters()) 
    SlowOpt = torch.optim.AdamW(hat_params, lr=SLOW_LR, weight_decay=1e-5)
    # SlowOpt = torch.optim.AdamW([{'params': filter(lambda p: p.requires_grad, model.parameters())}], lr=SLOW_LR,
    #                              weight_decay=1e-5)
    tail_params = list(decoder.parameters())
    FastOpt = torch.optim.AdamW(tail_params, lr=FAST_LR, weight_decay=1e-5)

    # Loss
    print("Creating loss function")
    loss_fn_token = torch.nn.NLLLoss()

    # Training Loop
    print("#" * 8)
    print("Training loop")
    os.makedirs("Checkpoints/Finetune", exist_ok=True)
    #os.makedirs(f"{args.checkpoint_path}\{name_of_project}\{name_of_run}", exist_ok=True)
    best_loss = 1e+10
    for epoch in range(EPOCHS):
        print("\nEpoch: ", epoch + 1)
        train(model, trainloader, SlowOpt, FastOpt, loss_fn_token, decoder)
        val_loss = validate(model, valloader, loss_fn_token, decoder)
        if val_loss < best_loss:
            torch.save(model, f"Checkpoints/Finetune/{name_of_project}-{name_of_run}.pt")
            best_loss = val_loss
            
    wandb.finish()

# main("asd-ssl", "lstm-base")
