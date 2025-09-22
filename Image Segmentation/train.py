import datetime
import gc
import time
from dataset import build_ade20k_loader
from config import Config
import torch 
from model import return_deeplabv3
import os, platform
from torch.amp import GradScaler,autocast
from utils import cos_weights,decay,get_ifgsm
import numpy as np
import random
import torch.backends.cudnn as cudnn
from loss import CombinedLoss

def init_setup(SEED):
    seed = SEED
    os.system('clear' if platform.system == "Linux" else 'cls')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
  
#NAN issues with autocast, so had to comment it out.
class Trainer:
    def __init__(self,config,model,task_criterion,dist_criterion,train_loader,val_loader,ifgsm_type,epochs,warmup_epochs,iters,target_layers,weighted_feats=True) -> None:
        self.device = 'cuda'
        self.config = config
        self.model = model
        self.model.to(self.device)

        self.target_layers = target_layers
        self.fmap = [None]*len(self.target_layers)
        
        self.num_layers = len(self.target_layers)
        self.hooks = [None]*len(self.target_layers)
        self.hook_flag = False

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=config.LR,weight_decay=cfg.weight_decay)
        self.min_loss = 1e+3
        self.loss_scaler = GradScaler(device=self.device)
        self.device = torch.device("cuda")
        self.task_criterion = task_criterion
        self.dist_criterion = dist_criterion
        self.warmup_epochs = warmup_epochs
        self.iters = iters
        self.epochs = epochs
        
        self.ifgsm_type = ifgsm_type
        self.lr = config.LR

        self.task_loss_list = torch.zeros(epochs)
        self.fmap_loss_list = torch.zeros(epochs)
        self.total_loss_list = torch.zeros(epochs)

        if weighted_feats:
            self.path = f"{config.out_dir}\\{ifgsm_type}_E{epochs}_WE{warmup_epochs}_Iters{iters}_weighted"
            os.makedirs(self.path, exist_ok=True)
            self.ckpt_path = self.path + "\\last.pt"
            self.best_ckpt_path = self.path + '\\best.pth'
            self.results_file = self.path + '\\results.txt'
            self.weights = cos_weights(num_layers=self.num_layers)
        else:
            self.path = f"{config.out_dir}\\{ifgsm_type}_E{epochs}_WE{warmup_epochs}_Iters{iters}"
            os.makedirs(self.path, exist_ok=True)
            self.ckpt_path = self.path + "\\last.pt"
            self.best_ckpt_path = self.path + '\\best.pth'
            self.results_file = self.path + '\\results.txt'
            self.weights = [1] * self.num_layers


        if os.path.exists(self.ckpt_path):
                print("Loading ckpt")
                self._load_ckpt(self.ckpt_path)
        else:
            for i in range(self.num_layers):
                self.hooks[i] = self.register_hook(i)
            self.hook_flag = True
        
        self.save_interval = self.config.SAVE_FREQ
        self.start_epoch = self.config.START_EPOCH

    def _save_ckpt(self, epoch, min_loss=False):
        
        for i in range(len(self.target_layers)):
            self.hooks[i].remove() 
        self.hook_flag = False
        ckpt = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
                'min_loss': self.min_loss,
                "task_loss":self.task_loss_list,
                "fmap_loss":self.fmap_loss_list,
                "total_loss":self.total_loss_list,
                "train_time":self.train_time.sum(),
                "total_time":self.test_time.sum()+self.train_time.sum(),
                "METRICS":{
                    'Accuracy':self.ac,
                    'Precision':self.pr,
                    'Recall':self.re,
                    'F1-Score':self.f1,
                    "Test_time":self.test_time.sum()
                        }
                }
        
        torch.save(ckpt, self.ckpt_path)
        print(f"Epoch {epoch+1} | Training ckpt saved at {self.ckpt_path}")
        if min_loss:
            torch.save(ckpt, self.best_ckpt_path)
            print(f"Epoch {epoch+1} | Best training ckpt saved at {self.best_ckpt_path}\n")

    def _load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
        self.model.load_state_dict(ckpt["model"], strict=False)
        for i in range(self.num_layers):
                self.hooks[i] = self.register_hook(i)
        self.hook_flag = True
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.start_epoch = ckpt['epoch'] + 1
        self.min_loss = ckpt['min_loss']
        if 'scaler' in ckpt:
            self.loss_scaler.load_state_dict(ckpt['scaler'])
        print(f"=> loaded successfully '{ckpt_path}' (epoch {ckpt['epoch']+1})")                
        del ckpt
        torch.cuda.empty_cache()

    def register_hook(self,index):
        def forward_hook(module,input,output):
            self.fmap[index] = output
        hook = self.target_layers[index].register_forward_hook(forward_hook)
        return hook

    def _run_epoch(self,epoch):
        if not self.hook_flag:
            for i in range(self.num_layers):
                self.hooks[i] = self.register_hook(i)
        ALPHA = decay(epoch, self.warmup_epochs, self.epochs)
        num_steps = len(self.train_loader)
        loss_fmap = 0.        
        task_loss_epoch = torch.zeros(num_steps)
        fmap_loss_epoch = torch.zeros(num_steps)

        if epoch >= self.warmup_epochs:
            total_loss_epoch = torch.zeros(num_steps)
        
        train_start = time.time()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        for idx, (xb, yb) in enumerate(self.train_loader):
    
            xb = xb.cuda(non_blocking=True)
            yb = yb.cuda(non_blocking=True)
            
            # with autocast(device_type='cuda', enabled=True):
            out = self.model(xb)['out']
            fmaps = self.fmap

            self.fmap = [None] * self.num_layers
            loss_task,_ = self.task_criterion(out,yb)
            
            task_loss_epoch[idx] = loss_task.item()

            if epoch >= self.warmup_epochs:
                self.model.eval()
                xb.requires_grad = True
                perturbed_dset = xb
                # with autocast(device_type='cuda', enabled=True):
                loss,_ = self.task_criterion(self.model(perturbed_dset)['out'], yb)
                self.fmap = [None] * self.num_layers
                # self.loss_scaler.scale(loss).backward(retain_graph=True)
                loss.backward(retain_graph = True)
                inv_fgsm = get_ifgsm(self.ifgsm_type, self.lr)                    
                
                for iteration in range(self.iters):
                    perturbed_dset = inv_fgsm(img = perturbed_dset, loss=loss,iteration = iteration)
                    out_new = self.model(perturbed_dset)['out']
                    loss,_ = self.task_criterion(out_new, yb)
                    
                self.model.train()
                out_new = self.model(perturbed_dset)['out']
                fmaps_new = self.fmap
                self.fmap = [None] * self.num_layers
                loss_fmap = 0.
                for i in range(self.num_layers):
                    loss_fmap = loss_fmap + (self.weights[i] * self.dist_criterion(fmaps_new[i].detach(), fmaps[i]))
                loss_total = (ALPHA * loss_task) + ((1-ALPHA)*loss_fmap)
                # self.loss_scaler.scale(loss_total).backward(retain_graph=True)
                loss_total.backward(retain_graph=True)
                # self.loss_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                # self.loss_scaler.step(self.optimizer)
                self.optimizer.step()
                # self.loss_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                fmap_loss_epoch[idx] = loss_fmap.item()
                total_loss_epoch[idx] = loss_total.item()
                print(f'\rTRAIN: EPOCH: [{epoch+1}/{self.epochs}] || iter [{idx+1}/{num_steps}]|| TASK LOSS: {loss_task.item():.3e} || FMAP LOSS: {loss_fmap.item():.3e} || TOTAL LOSS : {loss_total.item():.3e}',end="\r")
            else:
                # with autocast(device_type='cuda', enabled=True):
                loss_total = loss_task
                # self.loss_scaler.scale(loss_total).backward(retain_graph=True)
                loss_total.backward(retain_graph=True)
                # self.loss_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                # self.loss_scaler.step(self.optimizer)
                self.optimizer.step()
                # self.loss_scaler.update()
                self.optimizer.zero_grad()

                print(f'\rTRAIN: EPOCH: [{epoch+1}/{self.epochs}] || iter [{idx+1}/{num_steps}]|| TASK LOSS: {loss_task.item()}',end="\r")
                
            torch.cuda.empty_cache()
            _ = gc.collect()

        self.task_loss_list[epoch] = task_loss_epoch.mean()
        self.fmap_loss_list[epoch] = fmap_loss_epoch.mean()
        if epoch >= self.warmup_epochs:
            self.total_loss_list[epoch] = total_loss_epoch.mean()
        else:
            self.total_loss_list[epoch] = self.task_loss_list[epoch]
        self.train_time[epoch] = time.time() - train_start

        return self.task_loss_list[epoch].item(),self.fmap_loss_list[epoch].item(),self.total_loss_list[epoch].item()
    
    @torch.no_grad()
    def validate(self,epoch):
        self.loss_og = 0.
        self.fmap = [None] * self.num_layers
        self.model.eval()
        test_start = time.time()

        for _,(xb,yb) in enumerate(self.val_loader):            
            xb = xb.cuda(non_blocking = True)
            yb = yb.cuda(non_blocking = True)

            out_og = self.model(xb)['out']
            self.fmap = [None] * len(self.target_layers)
            loss,_ = self.task_criterion(out_og,yb)
            self.loss_og += loss
    
            torch.cuda.empty_cache()
            _ = gc.collect()

        self.test_time[epoch] = time.time() - test_start
        
        print(f"\n------------METRICS------------")
        print(f"Train Loss = {self.total_loss_list[epoch]:.3e}")
        print(f"Val Loss = {(self.loss_og/len(self.val_loader)):.3e}")
        print(f"Train time = {self.train_time[epoch]:.2f} s")
        print(f"Val time = {self.test_time[epoch]:.2f} s")
        print(f"Total time = {(self.train_time[epoch] + self.test_time[epoch]):.2f} s")
        print()
        return self.loss_og/len(self.val_loader)

    def train(self):
        print("Start training")
        start_time = time.time()
        self.train_time = torch.zeros(self.epochs)
        self.test_time = torch.zeros(self.epochs)
        for epoch in range(self.start_epoch, self.config.epochs):
            task,dist,total = self._run_epoch(epoch)
            val_loss = self.validate(epoch)
            torch.cuda.empty_cache()
            _ = gc.collect()
            print()
            if val_loss <= self.min_loss:
                min_loss = True
                self.min_loss = val_loss
            else:
                min_loss = False
            if (epoch % self.config.SAVE_FREQ == 0 or epoch == (self.config.epochs - 1)):
                self._save_ckpt(epoch, min_loss)
            with open(self.results_file, 'a') as f:
                f.write(f'Epoch: {epoch}\tTask Loss: {task}\tDist Loss: {dist}\tTotal Loss: {total}\tVal Loss: {val_loss}\n')
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'\nTraining time {total_time_str}')

def load_train_objs(config):
    data_loader_train, data_loader_val = build_ade20k_loader()
    print(f"Loading training objects\n")
    model = return_deeplabv3(num_classes=cfg.num_classes,backbone=cfg.backbone)
    # task_criterion = torch.nn.CrossEntropyLoss()
    task_criterion = CombinedLoss(ignore_index=cfg.ignore_index)
    dist_criterion = torch.nn.MSELoss()
    print(f"Training objects loaded\n")
    return data_loader_train, data_loader_val, model, task_criterion,dist_criterion

def main(config,ifgsm_type,epochs,warmup_rate,iters,weighted_feats):
    assert (warmup_rate <= 1) and (warmup_rate >= 0)
    warmup_epochs = int(warmup_rate * epochs)
    init_setup(config.seed)
    os.makedirs(config.out_dir, exist_ok=True) 
    train_loader, val_loader, model, task_criterion,dist_criterion = load_train_objs(config)
    target_layers = [getattr(model.backbone,name) for name in ['layer1','layer2','layer3','layer4']]
    trainer = Trainer(config,model,task_criterion,dist_criterion,train_loader,val_loader,ifgsm_type,epochs,warmup_epochs,iters,target_layers,weighted_feats=weighted_feats)
    trainer.train()

if __name__ == '__main__':
    cfg = Config()
    ifgsm_types = ['sgd', 'adam', 'ademamix']
    epoch_list = [cfg.epochs]
    # warmup_rates = [1.,0.75,0.5,0.25,0.]
    warmup_rates = [0.,0.25,0.5,0.75,1.]
    iteration_list = [5,10]
    batch_size = cfg.batch_size
    for epochs in epoch_list:
        for warmup_rate in warmup_rates:
            for iters in iteration_list:
                for ifgsm_type in ifgsm_types:
                    main(cfg,ifgsm_type=ifgsm_type, epochs=epochs, warmup_rate=warmup_rate, iters=iters, weighted_feats=True)
                    main(cfg,ifgsm_type=ifgsm_type, epochs=epochs, warmup_rate=warmup_rate, iters=iters, weighted_feats=False)
