from __future__ import annotations

import torch
import torch.nn.functional as F


def build_feature_bank(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            h = F.normalize(encoder(images.to(device)), dim=1)
            feats.append(h.cpu())
            labels.append(y.cpu())
    return torch.cat(feats, 0).T.contiguous().to(device), torch.cat(labels, 0).to(device)


def knn_accuracy(encoder, train_loader, test_loader, classes, device, k=200, temperature=0.1):
    bank, bank_labels = build_feature_bank(encoder, train_loader, device)
    encoder.eval()
    correct = total = 0
    k = min(k, bank.size(1))
    with torch.no_grad():
        for images, labels in test_loader:
            feat = F.normalize(encoder(images.to(device)), dim=1)
            sim = feat @ bank
            weight, idx = sim.topk(k=k, dim=1)
            lbl = bank_labels[idx]
            scores = torch.zeros(feat.size(0), classes, device=device)
            scores.scatter_add_(1, lbl, (weight / temperature).exp())
            pred = scores.argmax(1)
            correct += (pred.cpu() == labels).sum().item()
            total += labels.numel()
    return correct / total
