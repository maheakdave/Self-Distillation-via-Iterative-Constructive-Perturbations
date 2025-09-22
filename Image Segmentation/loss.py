import torch
import torch.nn as nn
import torch.nn.functional as F

def multiclass_dice_loss(pred, targets, ignore_index=255, smooth=1.0,temperature=3):
    """
    pred:   [N, C, H, W] raw logits
    targets: [N, 1, H, W] long tensor, possibly containing ignore_index
    """
    t = targets.squeeze(1).long()         
    valid = (t != ignore_index)           

    t_clamped = t.clone()
    t_clamped[~valid] = 0                 

    probs = F.softmax(pred/temperature, dim=1)       
    C = probs.shape[1]

    t_ohe = F.one_hot(t_clamped, num_classes=C)    
    t_ohe = t_ohe.permute(0, 3, 1, 2).float()      

    valid = valid.unsqueeze(1)            
    probs = probs * valid                 
    t_ohe  = t_ohe  * valid

    dims = (2, 3) 
    intersection = (probs * t_ohe).sum(dims)        
    cardinality  = probs.sum(dims) + t_ohe.sum(dims)  
    dice_per_class = (2. * intersection + smooth) / (cardinality + smooth)

    return 1.0 - dice_per_class.mean()



class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, weight_uq=1.0, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_uq = weight_uq
      
    def forward(self, outputs, targets):
        loss_cls = self.ce_loss(outputs, targets.squeeze(1).long())
        loss_cls = self.weight_ce * loss_cls
        loss_dice = multiclass_dice_loss(outputs, targets)
        loss_dice = self.weight_dice * loss_dice
        total_loss = (loss_cls + loss_dice)
        
        loss_dict = {
            "loss_cls": loss_cls.item(),
            "loss_dice": loss_dice.item(),
        }

        return total_loss, loss_dict
