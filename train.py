
import tqdm
import torch

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def train(iteration, optimizer):
    avg_loss = 0
    preds = []
    targets = []
    for j, (img,target,fix, dur) in enumerate(tqdm(trainloader)):
        if len(img) < batch_size:
            continue
        img, target, fix, dur = Variable(img), Variable(target.type(torch.FloatTensor)), Variable(fix,requires_grad=False), Variable(dur.type(torch.FloatTensor), requires_grad=False)
        if torch.cuda.is_available():
          img, target, fix, dur = img.cuda(), target.cuda(), fix.cuda(), dur.cuda()
        optimizer.zero_grad()

        pred = model(img,fix, dur)
        loss = F.binary_cross_entropy(pred,target)
        loss.backward()
        if clip != -1:
            clip_gradient(optimizer,clip)
        optimizer.step()
        avg_loss = (avg_loss*np.maximum(0,j) + loss.data.cpu().numpy())/(j+1)

        if j%25 == 0:
            wandb.log({'bce loss': avg_loss}, step=iteration)
        iteration += 1

        preds.append(pred.cpu())
        targets.append(target.to(torch.int16).cpu())
    with torch.no_grad():
      preds = torch.cat(preds, 0)
      targets = torch.cat(targets, 0)
      acc = accuracy(preds, targets)
      auc_v = auc(preds, targets, reorder=True)
      pre, rec = precision_recall(preds, targets)
      score = f1_score(preds, targets)
      print(f'\nT {epoch}: acc = {acc.item():.2f} auc = {auc_v.item():.2f} pre = {pre.item():.2f} rec = {rec.item():.2f} f1_score = {score.item():.2f}')

    return iteration