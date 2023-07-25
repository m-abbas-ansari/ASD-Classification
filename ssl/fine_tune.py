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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from model import BarlowTwins as SSL
from data import read_ASD_dataset, ASDHATDataset as Dataset, loo_split, image_selection

IMG_HEIGHT = 320
IMG_WIDTH = 512
HIDDEN_DIM = 256
LR = 1e-4
ANNO_DIR = "../../Data/Saliency4ASD/TrainingData"
IMG_DIR = "../../Data/Saliency4ASD/TrainingData/Images"
SEED = 42
VAL_RATIO = 0.2
FOLDS = 28
NUM_FIX = 20
BATCH_SIZE = 32
BACKEND="resnet18"
SELECT_NUMBER=100
EPOCHS = 10
SLOW_LR = 1e-5
FAST_LR = 1e-4
CLIP = 10

# Set Seeds for reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

transform = transforms.Compose([transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


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

def train(model, trainloader, SlowOpt, FastOpt, loss_fn_token, decoder, fold, iteration):
    token_losses = 0
    
    model.train()
    decoder.train()
    with tqdm(trainloader, unit='batch') as tepoch:
        minibatch = 0
        for i, data in enumerate(tepoch):
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
            out_token = torch.sigmoid(decoder(out)) 

            loss = F.binary_cross_entropy(out_token.squeeze(1),labels)
          
            loss.backward()
            token_losses += loss.item()

            if CLIP != -1:
                clip_gradient(SlowOpt, CLIP)
                clip_gradient(FastOpt, CLIP)
            
            # Update weights
            SlowOpt.step()
            FastOpt.step()
          
            # Log loss on wandb
            minibatch += 1.
            tepoch.set_postfix(token_loss=token_losses/minibatch)
            if i%25 == 0:
                wandb.log({'training loss_fold_'+str(fold+1):token_losses/minibatch})
            iteration += 1

    return iteration


# Validation Function
def validate(model, valloader, loss_fn_token, decoder, fold, epoch):
    avg_pred = []
    
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
                out_token = torch.sigmoid(decoder(out))  

                target = labels.data.cpu().numpy()[0]
                avg_pred.extend(out_token.data.cpu().numpy())

        # average voting
        avg_pred = np.mean(avg_pred)
        
        if not target:
            avg_pred = 1-avg_pred
        label = 'asd' if target else 'ctrl'
    
        wandb.log({'validation_acc_subject_' + label + '_' + str(fold+1): avg_pred})
            
    return avg_pred

def main(project, method="base", run="scratch"):
    print("#" * 8)
    print("Wandb setup")
    name_of_project = project
    name_of_run = "{}-{}".format(method, run)
    wandb.init(project=name_of_project, name=name_of_run)
    # print("Finetuning\nProject: {} Run: {}".format(name_of_project, name_of_run))
    # name_of_project = "ssl"
    # name_of_run = "fast-af-boi"

    # Load Data
    print("#" * 8)
    print("Loading data")
    anno = read_ASD_dataset(ANNO_DIR)
    preds = []
    targets = []
    
    for fold in range(FOLDS):
        
        train_data, val_data = loo_split(anno,fold)
        valid_id = image_selection(train_data, SELECT_NUMBER)
        
        train_set = Dataset(IMG_DIR,train_data,valid_id,NUM_FIX,IMG_HEIGHT,IMG_WIDTH,transform)
        val_set = Dataset(IMG_DIR,val_data,valid_id,NUM_FIX,IMG_HEIGHT,IMG_WIDTH,transform)
    

        print("Length of training data: ", len(train_set))
        print("Length of validation data: ", len(val_set))
        
        # Data Loaders
        print("#" * 8)
        print("Creating data loaders")
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


        # Model
        print("Creating model")
        model = torch.load(f"Checkpoints/ssl/lstm-BT-LARS/epoch-19.pt")
        # model = SSL(visual_backend=BACKEND, im_size=(IMG_HEIGHT, IMG_WIDTH), seq_len=NUM_FIX, hidden_dim=HIDDEN_DIM, batch_size=BATCH_SIZE)
        total_params = sum([p.numel() for p in filter(lambda p: p.requires_grad, model.parameters())])
        print(f"Total trainable parameters in model: {total_params:,}")

        if torch.cuda.is_available():
            model = model.cuda()
            print("Model on current device: ", torch.cuda.current_device())
        else:
            print('CUDA is not available. Model on CPU')

        decoder = nn.Linear(HIDDEN_DIM, 1).to("cuda")
        # decoder = nn.Sequential(*[nn.Linear(HIDDEN_DIM, HIDDEN_DIM), 
        #                         nn.ReLU(inplace=True),
        #                         nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        #                         nn.ReLU(inplace=True),
        #                         nn.Linear(HIDDEN_DIM, 2)]).to("cuda")
        
        # Optimizer
        print("Creating optimizer")
        hat_params = list(model.backbone.parameters()) 
        SlowOpt = torch.optim.AdamW(hat_params, lr=SLOW_LR, weight_decay=1e-5)
        # SlowOpt = torch.optim.AdamW([{'params': filter(lambda p: p.requires_grad, model.parameters())}], lr=SLOW_LR,
        #                              weight_decay=1e-5)
        tail_params = list(decoder.parameters())
        FastOpt = torch.optim.Adam(tail_params, lr=FAST_LR, weight_decay=1e-5)

        # Loss
        print("Creating loss function")
        loss_fn_token = torch.nn.NLLLoss()

        # Training Loop
        print("#" * 8)
        print('Start %d-fold validation for fold %d' %(FOLDS,fold+1))
        os.makedirs("Checkpoints/Finetune", exist_ok=True)
        #os.makedirs(f"{args.checkpoint_path}\{name_of_project}\{name_of_run}", exist_ok=True)
        best_acc = 0
        iteration = 0
        for epoch in range(EPOCHS):
            print("\nEpoch: ", epoch + 1)
            adjust_lr(SlowOpt, epoch)
            adjust_lr(FastOpt, epoch)
            iteration = train(model, trainloader, SlowOpt, FastOpt, loss_fn_token, decoder, fold, iteration)
            acc = validate(model, valloader, loss_fn_token, decoder, fold, epoch)
            if acc > best_acc:
                full_model = nn.Sequential(*[model.backbone, decoder])
                torch.save(full_model, f"Checkpoints/Finetune/{name_of_project}-{name_of_run}-{fold+1}.pt")
                best_acc = acc
          
        del model 
        
        # Get metrics for best model on current fold
        m = torch.load(f"Checkpoints/Finetune/{name_of_project}-{name_of_run}-{fold+1}.pt")
        all_scores = []
        with torch.no_grad(): 
            for data in valloader:
                imgs, fixs, _, pad_mask, labels = data

                if torch.cuda.is_available():  # Move to GPU if available
                    fixs = fixs.cuda()
                    pad_mask = pad_mask.cuda()
                    labels = labels.cuda().squeeze(1)
                    imgs = imgs.cuda()

                pred = torch.sigmoid(m[1](m[0](imgs, fixs, pad_mask)))
                all_scores.extend(pred.squeeze(1).data.cpu().numpy())
        preds.append(np.mean(all_scores))
        targets.append(labels[0].item())
    
    
    print("#"*20)
    print("Performing final evaluation")

    pred_scores = preds
    preds = (np.array(preds) > 0.5).astype(float)
    
    auc = roc_auc_score(targets, pred_scores)
    acc = accuracy_score(targets, preds)         
    cm = confusion_matrix(targets, preds)
    specificity = cm[0, 0] / sum(cm[0,:]) # TN / (TN + FP)
    sensitivty = cm[1, 1] / sum(cm[1, :]) # TP / (TP + FN)
    
    print(
        "Acc: {:.4f}, Sens: {:.4f}, Spec: {:.4f}, AUC: {:.4f}".format(acc, 
                                                                      sensitivty,
                                                                      specificity,
                                                                      auc))
    wandb.log({
        "Final Acc": acc,
        "Final Sens": sensitivty,
        "Final Spec": specificity,
        "Final AUC": auc
    })
    wandb.finish(quiet=True)
    
main("asd-loo-new", "bt-lars")
