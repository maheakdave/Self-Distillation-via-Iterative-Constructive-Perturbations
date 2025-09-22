import torch
import torch.nn as nn
import math
from typing import Tuple,OrderedDict,List
import warnings

class AdamInvFGSM(nn.Module):
    def __init__(self,lr,wd=0,beta1=0.9,beta2=0.999,eps = 1e-8,amsgrad = False):
        super().__init__()

        """
        https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        """

        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0
        self.v = 0
        self.wd = wd
        self.eps = eps
        self.lr = lr
        self.vcapmax = torch.tensor(0.)
        self.amsgrad = amsgrad
    def forward(self,img,loss,iteration):
        grad_data = torch.autograd.grad(outputs=loss,inputs=img)[0]
        if self.wd:
            grad_data = grad_data + (self.wd*img)
        self.m = (self.beta1*self.m) + ((1-self.beta1)*grad_data)
        self.v = (self.beta2*self.v) + ((1-self.beta2)*grad_data.pow(2))
        mcap = self.m/(1-math.pow(self.beta1,iteration+1))
        vcap = self.v/(1-math.pow(self.beta2,iteration+1))
        if self.amsgrad:
            vcap = torch.maximum(self.vcapmax,vcap)
            img = img - self.lr*mcap/(vcap.pow(0.5)+self.eps)
            return img    
        img = img - self.lr*mcap/(vcap.pow(0.5)+self.eps)
        return img

class AdemamixInvFGSM(nn.Module):
    def __init__(self,lr,wd=0,beta1=0.9,beta2=0.999,beta3 = 0.9999,eps = 1e-8,alpha = 5.0):
        super().__init__()

        """
        https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch?tab=readme-ov-file
        """

        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.m1 = 0
        self.m2 = 0
        self.v = 0
        self.wd = wd
        self.eps = eps
        self.lr = lr
        self.alpha = alpha
        self.vcapmax = torch.tensor(0.)
    def forward(self,img,loss,iteration):
        grad_data = torch.autograd.grad(outputs=loss,inputs=img,retain_graph=True)[0]
        # if self.wd:
        #     grad_data = grad_data + (self.wd*img)
        self.m1 = (self.beta1*self.m1) + ((1-self.beta1)*grad_data)
        self.m2 = (self.beta3*self.m2) + ((1-self.beta3)*grad_data)
        self.v = (self.beta2*self.v) + ((1-self.beta2)*grad_data.pow(2))
        mcap = self.m1/(1-math.pow(self.beta1,iteration+1))   
        vcap = self.v/(1-math.pow(self.beta2,iteration+1))  
        img = img - (self.lr*((mcap+(self.alpha*self.m2))/(vcap.pow(0.5)+self.eps)+self.wd*img))

class SGDInvFGSM(nn.Module):
    def __init__(self,lr):
        super().__init__()
        self.lr = lr
        
    def forward(self,img,loss,iteration):
        #NO use of iteration parameter, just there to keep out the iteration error in the training loop.
        grad_data = torch.autograd.grad(outputs=loss,inputs=img)[0]
        img = img - (self.lr*grad_data)
        return img
    
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def timer(t):
    if round(t,1) < 60:
        sec = round(t)
        if sec < 10:
            t2 = f'00:00:0{sec}'
        else:
            t2 = f'00:00:{sec}'
    elif (t >= 60) and (t < 3600):
        min = round((math.modf((t)/60))[1])
        sec = round((math.modf((t)/60))[0]*60)
        if sec < 10 and min < 10:
            t2 = f'00:0{min}:0{sec}'
        elif sec < 10 and min >= 10:
            t2 = f'00:{min}:0{sec}'
        elif sec >= 10 and min < 10:
            t2 = f'00:0{min}:{sec}'
        elif sec >= 10 and min >= 10:
            t2 = f'00:{min}:{sec}'
    else:
        hour = round((math.modf((t)/3600))[1])
        min = round((math.modf((t)/3600))[0]*60)
        sec = round((math.modf((math.modf((t)/3600))[0]*60))[0]*60)
        if hour < 10:
            if min < 10:
                if sec < 10:
                    t2 = f'0{hour}:0{min}:0{sec}'
                else:
                    t2 = f'0{hour}:0{min}:{sec}'
            else:
                if sec < 10:
                    t2 = f'0{hour}:{min}:0{sec}'
                else:
                    t2 = f'0{hour}:{min}:{sec}'
        else:
            if min < 10:
                if sec < 10:
                    t2 = f'{hour}:0{min}:0{sec}'
                else:
                    t2 = f'{hour}:0{min}:{sec}'
            else:
                if sec < 10:
                    t2 = f'{hour}:{min}:0{sec}'
                else:
                    t2 = f'{hour}:{min}:{sec}'
    return t2

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cos_weights(num_layers, lowest_weight=math.cos(math.pi/4)):
    return [math.cos((i / (num_layers - 1)) * math.acos(lowest_weight)) for i in range(num_layers)]

def cos_weights_inverse(num_layers, highest_weight=math.cos(math.pi/4)):
    return [math.cos(((num_layers - 1 - i)/ (num_layers-1)) * math.acos(highest_weight)) for i in range(num_layers)]

def decay(epoch, warmup_epochs, epochs):
    if epoch>warmup_epochs:
        return math.cos(((math.pi*(epoch-warmup_epochs))/((2*epochs)-(2*warmup_epochs))))
    return 1

def get_ifgsm(ifgsm_type: str, lr: float = 2e-3):
    if ifgsm_type.lower() == 'adam':
        inv_fgsm = AdamInvFGSM(lr=lr)
    elif ifgsm_type.lower() == 'ademamix':
        inv_fgsm = AdemamixInvFGSM(lr=lr)
    elif ifgsm_type.lower() == 'sgd':
        inv_fgsm = SGDInvFGSM(lr=lr)
    else:
        NotImplementedError(f'{ifgsm_type.lower()} not implemented... Please select adam, ademamix or sgd...')
    return inv_fgsm

def combine_loss(pred):
    total_loss = 0
    for key in pred.keys():
            total_loss += pred[key]
    return total_loss

def freeze_layer(layer,unfreeze=False):
    for param in layer.parameters():
        if not unfreeze:
            param.requires_grad = False
        else:
            param.requires_grad = True
