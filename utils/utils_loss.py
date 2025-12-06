import os
from typing import Type
import torch
import torch.nn.functional as F
import pandas as pd
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm

def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

def soft_cross_entropy(logits, soft_targets, temperature=1.0):
    """
    软交叉熵损失函数
    Args:
        logits: [batch_size, num_classes] - 未归一化的预测logits
        soft_targets: [batch_size, num_classes] - 软标签概率分布
        temperature: 温度参数
    """
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    loss = -torch.sum(soft_targets * log_probs, dim=-1)
    return loss.mean()

   
def clip_loss(x, y, temperature=0.07, device='cuda'):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    sim = torch.einsum('i d, j d -> i j', x, y) * 1 / temperature

    labels = torch.arange(x.shape[0]).to(device)

    loss_t = F.cross_entropy(sim, labels) 
    loss_i = F.cross_entropy(sim.T, labels) 

    i2t_acc1, i2t_acc5 = precision_at_k(
        sim, labels, top_k=(1, 5))
    t2i_acc1, t2i_acc5 = precision_at_k(
        sim.T, labels, top_k=(1, 5))
    acc1 = (i2t_acc1 + t2i_acc1) / 2.
    acc5 = (i2t_acc5 + t2i_acc5) / 2.

    return (loss_t + loss_i), acc1, acc5


def adaptive_feature_clip_loss(x, y, temperature=0.07, soft_temperature=0.1, alpha=0.3, device='cuda'):
    """
    自适应混合硬标签和特征相似度软标签
    
    Args:
        alpha: 硬标签权重 [0, 1]
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    
    sim = torch.einsum('i d, j d -> i j', x, y) / temperature
    
    # 硬标签损失（原始CLIP）
    hard_labels = torch.arange(x.shape[0]).to(device)
    hard_loss_i2t = F.cross_entropy(sim, hard_labels)
    hard_loss_t2i = F.cross_entropy(sim.T, hard_labels)
    
    # 软标签损失（基于特征相似度）
    with torch.no_grad():
        # 使用模态内相似度
        y_self_sim = torch.einsum('i d, j d -> i j', y, y)  # [B, B]
        
        # 生成软标签
        soft_labels_i2t = F.softmax(y_self_sim / soft_temperature, dim=-1)  # ECG → 文本
        soft_labels_t2i = F.softmax(y_self_sim.T / soft_temperature, dim=-1)  # 文本 → ECG
    
    soft_loss_i2t = soft_cross_entropy(sim, soft_labels_i2t)
    soft_loss_t2i = soft_cross_entropy(sim.T, soft_labels_t2i)
    
    # 混合损失
    loss_i2t = alpha * hard_loss_i2t + (1 - alpha) * soft_loss_i2t
    loss_t2i = alpha * hard_loss_t2i + (1 - alpha) * soft_loss_t2i
    
    # 计算准确率
    i2t_acc1, i2t_acc5 = precision_at_k(sim, hard_labels, top_k=(1, 5))
    t2i_acc1, t2i_acc5 = precision_at_k(sim.T, hard_labels, top_k=(1, 5))
    acc1 = (i2t_acc1 + t2i_acc1) / 2.
    acc5 = (i2t_acc5 + t2i_acc5) / 2.
    
    return (loss_i2t + loss_t2i), acc1, acc5