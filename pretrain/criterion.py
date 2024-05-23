import torch
from torch import nn


class NCELoss(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q):
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.temperature)
        out = out.contiguous()

        loss = self.criterion(out, target)
        return loss

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, k, q, labels):
        # 计算所有样本对之间的相似度
        similarity_matrix = torch.mm(k, q.T)
        # 应用温度参数
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)
        
        # 创建标签矩阵，用于标识正样本和负样本
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()

        # 计算损失
        positive_samples = similarity_matrix * mask
        negative_samples = similarity_matrix * (1 - mask)
        
        # 避免自身对比 (ok for different views)
        # positive_samples -= torch.eye(*positive_samples.shape, device=positive_samples.device)
        
        # 计算最终损失
        positive_sum = positive_samples.sum(dim=1)
        negative_sum = negative_samples.sum(dim=1)
        
        losses = -torch.log(positive_sum / (positive_sum + negative_sum))
        return losses.mean()
