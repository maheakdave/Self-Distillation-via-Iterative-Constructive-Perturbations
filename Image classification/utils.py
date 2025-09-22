import torch
import torch.nn as nn
import math
from torchvision.models import resnet50
import copy

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

voc_object_categories = {0:'aeroplane',1: 'bicycle',2: 'bird',3: 'boat',
                     4:'bottle',5: 'bus',6: 'car',7: 'cat',8: 'chair',
                     9:'cow',10: 'diningtable',11: 'dog',12: 'horse',
                     13:'motorbike',14: 'person',15: 'pottedplant',
                     16:'sheep',17: 'sofa',18: 'train',19: 'tvmonitor'}

def modify_resnet50(model):
    model.conv1.kernel_size = 3
    model.conv1.padding =1
    model.maxpool = nn.Identity()
    return model

class ResNetSelfDistillationWrapper(torch.nn.Module):
    def __init__(self,model,target_layers):
        super().__init__()
        self.model = model
        self.activations = [None]*len(target_layers)
        self.soft_activations = [None]*(len(target_layers)-1)
        self.target_layers = target_layers
        self.bn1 = torch.nn.Sequential(
            copy.deepcopy(target_layers[1]),
            copy.deepcopy(target_layers[2]),
            copy.deepcopy(target_layers[3]),
        )
        self.fc1 =torch.nn.Sequential(copy.deepcopy(self.model.avgpool),torch.nn.Flatten(),copy.deepcopy(self.model.fc))
        self.bn2 = torch.nn.Sequential(
            copy.deepcopy(target_layers[2]),
            copy.deepcopy(target_layers[3]),
        )
        self.fc2 = torch.nn.Sequential(copy.deepcopy(self.model.avgpool),torch.nn.Flatten(),copy.deepcopy(self.model.fc))
        self.bn3 = torch.nn.Sequential(
            copy.deepcopy(target_layers[3]),
        )
        self.fc3 = torch.nn.Sequential(copy.deepcopy(self.model.avgpool),torch.nn.Flatten(),copy.deepcopy(self.model.fc))
        self.fhooks = [self.register_hook(i) for i in range(len(target_layers))]

    def register_hook(self,idx):
        def forward_hook(module,input,output):
            self.activations[idx]=output
        return self.target_layers[idx].register_forward_hook(forward_hook)
    
    def forward(self,xb):
        soft_logits = [None]*3
        logits = self.model(xb)
        for i in range(len(self.target_layers)-1):
            self.soft_activations[i] = getattr(self,f'bn{i+1}')(self.activations[i])
            soft_logits[i] = getattr(self,f'fc{i+1}')(self.soft_activations[i])
        return soft_logits,logits,self.activations,self.soft_activations

# ConvNext wrapper is still work in progress
class ConvNextSelfDistillationWrapper(torch.nn.Module):
    def __init__(self,model,target_layers):
        super().__init__()
        self.model = model
        self.activations = [None]*len(target_layers)
        self.soft_activations = [None]*(len(target_layers)-1)
        self.target_layers = target_layers
        self.bn1 = torch.nn.Sequential(
            *[copy.deepcopy(target_layers[i]) for i in range(1,len(target_layers))],
        )
        self.fc1 =torch.nn.Sequential(copy.deepcopy(self.model.avgpool),torch.nn.Flatten(),copy.deepcopy(self.model.classifier))
        self.bn2 = torch.nn.Sequential(
            *[copy.deepcopy(target_layers[i]) for i in range(2,len(target_layers))],
        )
        self.fc2 = torch.nn.Sequential(copy.deepcopy(self.model.avgpool),torch.nn.Flatten(),copy.deepcopy(self.model.classifier))
        self.bn3 = torch.nn.Sequential(
            *[copy.deepcopy(target_layers[i]) for i in range(3,len(target_layers))],
        )
        self.fc3 = torch.nn.Sequential(copy.deepcopy(self.model.avgpool),torch.nn.Flatten(),copy.deepcopy(self.model.classifier))
        self.bn4 = torch.nn.Sequential(
            *[copy.deepcopy(target_layers[i]) for i in range(4,len(target_layers))],
        )
        self.fc4 = torch.nn.Sequential(copy.deepcopy(self.model.avgpool),torch.nn.Flatten(),copy.deepcopy(self.model.classifier))
        self.bn5 = torch.nn.Sequential(
            *[copy.deepcopy(target_layers[i]) for i in range(5,len(target_layers))],
        )
        self.fc5 = torch.nn.Sequential(copy.deepcopy(self.model.avgpool),torch.nn.Flatten(),copy.deepcopy(self.model.classifier))
        self.bn6 = torch.nn.Sequential(
            *[copy.deepcopy(target_layers[i]) for i in range(6,len(target_layers))],
        )
        self.fc6 = torch.nn.Sequential(copy.deepcopy(self.model.avgpool),torch.nn.Flatten(),copy.deepcopy(self.model.classifier))
        self.bn7 = torch.nn.Sequential(
            *[copy.deepcopy(target_layers[i]) for i in range(7,len(target_layers))],
        )
        self.fc7 = torch.nn.Sequential(copy.deepcopy(self.model.avgpool),torch.nn.Flatten(),copy.deepcopy(self.model.classifier))

        self.fhooks = [self.register_hook(i) for i in range(len(target_layers))]

    def register_hook(self,idx):
        def forward_hook(module,input,output):
            self.activations[idx]=output
        return self.target_layers[idx].register_forward_hook(forward_hook)
    
    def forward(self,xb):
        soft_logits = [None]*3
        logits = self.model(xb)
        for i in range(len(self.target_layers)-1):
            self.soft_activations[i] = getattr(self,f'bn{i+1}')(self.activations[i])
            soft_logits[i] = getattr(self,f'fc{i+1}')(self.soft_activations[i])
        return soft_logits,logits,self.activations,self.soft_activations

def distillation_loss(soft_logits,logits,activations,soft_activations,labels,alpha=0.1,beta=1e-6,T=3):

    l2_loss = sum([torch.nn.functional.mse_loss(sa,activations[-1],reduction='mean') for sa in soft_activations])
    logit_loss = sum([torch.nn.functional.kl_div(torch.log_softmax(sl/T,dim=1),torch.log_softmax(logits/T,dim=1),reduction='batchmean',log_target=True)*T*T for sl in soft_logits])
    label_loss = sum([torch.nn.functional.cross_entropy(sl,labels) for sl in soft_logits])
    task_loss = torch.nn.functional.cross_entropy(logits,labels)
    total_loss = ((1-alpha) * (label_loss/len(soft_logits))) + (alpha * (logit_loss/len(soft_logits))) + (beta * (l2_loss/len(soft_activations))) + task_loss

    return total_loss

def freeze_layer(layer,unfreeze=False):
    for param in layer.parameters():
        if not unfreeze:
            param.requires_grad = False
        else:
            param.requires_grad = True
