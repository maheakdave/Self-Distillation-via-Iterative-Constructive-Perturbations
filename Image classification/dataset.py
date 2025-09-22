from config import Config

def build_cifar100_loader():
    cfg = Config()
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=(int(cfg.ORIGINAL_SIZE[0]*cfg.ratio),int(cfg.ORIGINAL_SIZE[1]*cfg.ratio))),
        transforms.Resize(size=cfg.ORIGINAL_SIZE),
        transforms.RandAugment(num_ops=2, magnitude=9,interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])
    
    train_set = CIFAR100(root="CIFAR100",download=False,transform = train_transform,train=True)
    val_set = CIFAR100(root="CIFAR100",download=False,transform = val_transform,train=True)
    
    train_loader = DataLoader(train_set,batch_size=cfg.BATCH_SIZE,shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_set,batch_size=cfg.BATCH_SIZE,shuffle=False,pin_memory=True)
    
    return train_loader,val_loader
