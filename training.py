import statistics
import time

from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
import torch


def evaluate(net, loader, device, verbose=True):
    start = time.time()
    
    all_logits = []
    all_labels = []
    all_losses = []

    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            logits = net(X.to(device))
            loss = net.loss_fn(logits, y.to(device))
            all_logits.extend(list(logits.cpu().numpy()))
            all_labels.extend(list(y))
            all_losses.append(loss.item())

    val_loss = statistics.mean(all_losses)
    auprc = average_precision_score(all_labels, all_logits)
    auroc = roc_auc_score(all_labels, all_logits)
    
    if verbose:
        print(f'Average precision score: {auprc}')
        print(f'AUROC: {auroc}')
        print(f'Validation loss (approximate): {val_loss}')
        print(f'Elapsed: {time.time() - start}')
    return val_loss, auprc, auroc

def train_epoch(net, loader, device, verbose=True):
    start = time.time()
    
    losses = []
    for X, y in loader:
        loss = net.train_step(X.to(device), y.to(device))
        loss = loss.item()
        losses.append(loss)
    train_loss = statistics.mean(losses)
    
    if verbose:
        print(f'Train loss (approximate): {train_loss}')
        print(f'Elapsed: {time.time() - start}')
    return train_loss
