import torch,os,math,random
from PIL import Image
from torchvision.transforms import transforms
from config import Config
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class ADE20KDataset(Dataset):
    def __init__(self, cfg, split="training", aug=True):
        self.root = cfg.data_path
        self.split = split
        self.aug = aug
        self.img_sz = cfg.img_size
        self.ignore_index = cfg.ignore_index
        
        self.image_dir = os.path.join(self.root, "images", split)
        self.annotation_dir = os.path.join(self.root, "annotations", split)

        self.images = sorted(os.listdir(self.image_dir))
        self.annotations = sorted(os.listdir(self.annotation_dir))
        self.photometric = transforms.Compose([
                                    transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=cfg.sharpness_prob),
                                    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=cfg.sharpness_prob),
                                    transforms.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast, saturation=cfg.saturation, hue=cfg.hue)])
        
        self.degrees = cfg.degrees
        self.translate = cfg.translate
        self.scale = cfg.scale
        self.val_img_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.mask_transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.images)

    def get_params(self, degrees, translate, scale_ranges, shears, img_size):
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def resize_to_fit(self,image, mask):
        img_width, img_height = image.size
        scale = max(self.img_sz / img_width, self.img_sz / img_height)
        new_width = math.ceil(img_width * scale)
        new_height = math.ceil(img_height * scale)
        resized_image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        resized_mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
        return resized_image, resized_mask

    def random_crop(self, image, mask):
        img_width, img_height = image.size
        if img_width < self.img_sz or img_height < self.img_sz:
            image, mask = self.resize_to_fit(image, mask)
        img_width, img_height = image.size
        top = random.randint(0, img_height - self.img_sz)
        left = random.randint(0, img_width - self.img_sz)
        crop_box = (left, top, left + self.img_sz, top + self.img_sz)
        cropped_image = image.crop(crop_box)
        cropped_mask = mask.crop(crop_box)
        return cropped_image, cropped_mask

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.annotation_dir, self.annotations[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        img_width, img_height = image.size
        scale = self.img_sz / min(img_width, img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)

        image, mask = self.random_crop(image, mask)
        if self.split == 'training' and self.aug:
            image = self.photometric(image)
            affine_params = self.get_params(degrees=self.degrees, translate=self.translate, scale_ranges=self.scale, shears=None, img_size=(self.img_sz, self.img_sz))
            image = F.affine(image,*affine_params, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = F.affine(mask,*affine_params, interpolation=transforms.InterpolationMode.NEAREST_EXACT, fill=self.ignore_index)
        return self.val_img_transform(image), self.mask_transform(mask)*255

def build_ade20k_loader():

    cfg = Config()
    train_set = ADE20KDataset(cfg=cfg, split="training")
    val_set = ADE20KDataset(cfg=cfg,split="validation")
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=cfg.batch_size,shuffle=True,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=cfg.batch_size,shuffle=False,pin_memory=True)

    return train_loader,val_loader
