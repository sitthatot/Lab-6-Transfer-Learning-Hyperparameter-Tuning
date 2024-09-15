import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset,Dataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
import cv2
from skimage.util import random_noise
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

seed = 4912
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


### START CODE HERE ###
class CustomImageDataset(Dataset):
    def __init__(self, image_paths,gauss_noise=False,gauss_blur=None,resize=128,p=0.5, center_crop=False, transform=None):
        self.p = p
        self.resize = resize
        self.gauss_noise = gauss_noise
        self.gauss_blur = gauss_blur
        self.center_crop = center_crop
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def add_gaussian_noise(self, image):
        #สร้าง gaussian noise
        x, y, channel = image.shape
        mean = random.uniform(-50, 50)
        gaussian_noise = np.random.normal(loc=mean,scale=100,size=(x,y,channel)).astype(np.float32)
        #ภาพ ผสมกับ Gaussian Noise
        noisy_image = image + gaussian_noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def do_gaussian_blur(self, image):
        kernel_size = random.randrange(3, 12, 2)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0) #ภาพ,Kernel Size, Standard Deviation
    
    def do_center_crop(self, image, desire_h=128, desire_w=128):
        height,width, _ = image.shape
        if width <= height: #กรณี width สั้น ให้ resize โดยยึด width 
            #ปรับ width เทียบเท่าแล้ว เหลือ Height ที่ยังเกิน
            #เพื่อให้สัดส่วนคงเดิม ไม่ถูกบีบ ให้หาอัตราส่วนภาพ
            img_ratio = height / width
            image = cv2.resize(image, (self.resize, int(self.resize*img_ratio))) #image , (width, height) ที่จะไป
            #พอ resize แล้ว จะได้ width height ตัวใหม่ 
            new_resize_h, new_resize_w, _ = image.shape
            crop_img = image[new_resize_h//2 - self.resize//2: new_resize_h//2 + self.resize//2, 0:self.resize]
        else:
            #ปรับ Height เทียบเท่าแล้ว เหลือ width ที่ยังเกิน
            img_ratio = width / height
            image = cv2.resize(image, (int(self.resize*img_ratio), self.resize))
            new_resize_h, new_resize_w, _ = image.shape
            crop_img = image[0:self.resize, new_resize_w//2 - self.resize//2: new_resize_w//2 + self.resize//2]
        return crop_img

    def __getitem__(self, idx):
        image_paths = self.image_paths[idx]
        img = plt.imread(image_paths)
        if img.ndim == 2: #ถ้าเป็น Gray --> บังคับเป็น RGB Format
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        #ถ้าทำ center crop ก็ไม่ต้องทำ Resize, เลือกอย่างใดอย่างนึง
        if(self.center_crop):
            gt_image = self.do_center_crop(img, desire_h=self.resize, desire_w=self.resize)
            # print("*******")
            # print(gt_image.shape)
        else:
            gt_image = cv2.resize(img, (self.resize, self.resize))  #Ground truth image
        image = gt_image.copy()

    # Ensure minimum size of 7x7
        if image.shape[0] < 7 or image.shape[1] < 7:
            image = cv2.resize(image, (max(7, image.shape[1]), max(7, image.shape[0])))
        if gt_image.shape[0] < 7 or gt_image.shape[1] < 7:
            gt_image = cv2.resize(gt_image, (max(7, gt_image.shape[1]), max(7, gt_image.shape[0])))
            
            
        #ใส่ความน่าจะเป็น ที่จะถูก Apply G-noise, G-blur
        if self.p >= 0.5:
            if self.gauss_noise:
                image = self.add_gaussian_noise(image)
            if self.gauss_blur:
                image = self.do_gaussian_blur(image)
        # print("===================")
        # print(f"Image shape: {image.shape}")
        if self.transform:
            image = self.transform(image)
            gt_image = self.transform(gt_image)
        return image, gt_image
### END CODE HERE ###

### START CODE HERE ###
data_dir = "data/img_align_celeba"
image_paths = []
for file_name in os.listdir(data_dir):
    image_paths.append(f"{data_dir}/{file_name}")
print(image_paths)

transform = transforms.Compose([transforms.ToTensor()])

dataset = CustomImageDataset(image_paths=image_paths,
                             gauss_noise=True,
                             gauss_blur=True,
                             resize=128,
                             p=0.5,
                             center_crop=True,
                             transform=transform
                            )
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
### END CODE HERE ###

batch,gt_img = next(iter(dataloader)) 
print(batch.shape)
torch.is_tensor(batch)

### START CODE HERE ###
class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DownSamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpSamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x


### START CODE HERE ###
class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DownSamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpSamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, architecture):
        super(Autoencoder, self).__init__()
        self.architecture = architecture
        #Encoder
        
        if len(self.architecture)  == 3:
            print("this is 3")
            print(type(self.architecture[0]))
            print("==================")
            print(int(self.architecture[0]))
            # self.conv_in = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            # self.down1 = DownSamplingBlock(64, 128, kernel_size=3, stride=1, padding=1)
            # ### DESIGN YOUR OWN MODEL ###
            # self.down2 = DownSamplingBlock(128, 256, kernel_size=3, stride=1, padding=1)
            # #Decoder
            # self.up3 = UpSamplingBlock(256, 128, kernel_size=3, stride=1, padding=1)
            # self.up4 = UpSamplingBlock(128, 64, kernel_size=3, stride=1, padding=1)
            # self.conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        elif len(self.architecture)  == 4:
            print("this is 4")
            print(type(self.architecture[0]))
            print("==================")
            print(int(self.architecture[0]))
            # print(self.architecture)
    #     self.conv_in = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    #     self.down1 = DownSamplingBlock(64, 128, kernel_size=3, stride=1, padding=1)
    #     ### DESIGN YOUR OWN MODEL ###
    #     self.down2 = DownSamplingBlock(128, 256, kernel_size=3, stride=1, padding=1)
    #     #Decoder
    #     self.up3 = UpSamplingBlock(256, 128, kernel_size=3, stride=1, padding=1)
    #     self.up4 = UpSamplingBlock(128, 64, kernel_size=3, stride=1, padding=1)
    #     self.conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    # def forward(self, x):
    #     x = self.conv_in(x)
    #     x = self.down1(x)
    #     x = self.down2(x)
        
    #     x = self.up3(x)
    #     x = self.up4(x)
    #     x = self.conv(x)
    #     return x
### END CODE HERE ###

from ray import tune
from ray.tune.schedulers import ASHAScheduler

def short_dirname(trial):
    return "trial_" + str(trial.trial_id)

def train_model(config):
    # Extract hyperparameters from config
    architecture = config["architecture"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    optimizer = config["optimizer"]

    model = Autoencoder(architecture)
    
    # Example training code
    print(f"Training with lr={lr} and batch_size={batch_size}, {architecture},{num_epochs},{optimizer}")

config = {
    'architecture': tune.choice([[32, 64, 128], [64, 128, 256], [64, 128, 256, 512]]),
    "lr": tune.grid_search([1e-3, 1e-4, 1e-2]),
    "batch_size": tune.grid_search([16, 32]),
    "num_epochs": tune.grid_search([10, 50, 100]),
    'optimizer': tune.choice(['Adam', 'SGD']),
}

tune.run(
    train_model,
    config=config,
    num_samples=2,  # Number of trials
    trial_dirname_creator=short_dirname,
)

# model = Autoencoder(architecture)