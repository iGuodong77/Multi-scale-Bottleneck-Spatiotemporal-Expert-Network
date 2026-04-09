import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [N, 2]  —— 你的 net 输出
        targets: [N]    —— 标签，0 或 1
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [N]
        pt = torch.exp(-ce_loss)  # pt = exp(-CE)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss - 直接优化与F1等价的Dice系数
    
    Dice = 2*TP / (2*TP + FP + FN) = F1
    最小化 DiceLoss = 1 - Dice 等价于最大化 F1
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        logits: [N, 2] - 网络输出
        targets: [N] - 标签 0 或 1
        """
        probs = F.softmax(logits, dim=1)[:, 1]  # 取正类（变化类）概率
        targets_float = targets.float()
        
        intersection = (probs * targets_float).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets_float.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """组合损失函数 = Focal Loss + Dice Loss
    
    - Focal Loss: 处理类别不平衡，关注难分类样本
    - Dice Loss: 直接优化F1分数
    
    组合使用可同时提升 Kappa 和 F1
    """
    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
    
    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return focal_loss + self.dice_weight * dice_loss