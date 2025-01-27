import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from glob import glob
import pandas as pd
import sys
import csv
import lpips
import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import progressbar
import timeit
start = timeit.default_timer()
class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))
    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()
class DISTS(torch.nn.Module):
    def __init__(self, load_weights=True):
        super(DISTS, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])   
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))
        self.chns = [3,64,128,256,512,512]
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.alpha.data.normal_(0.1,0.01)
        self.beta.data.normal_(0.1,0.01)
        if load_weights:
            weights = torch.load('weights_DISTS.pt')
            self.alpha.data = weights['alpha']
            self.beta.data = weights['beta']   
    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
    def forward(self, x, y, require_grad=False, batch_average=False):
        if require_grad:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)   
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y) 
        dist1 = 0 
        dist2 = 0 
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha/w_sum, self.chns, dim=1)
        beta = torch.split(self.beta/w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2,3], keepdim=True)
            y_mean = feats1[k].mean([2,3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)
            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)
        score = 1 - (dist1+dist2).squeeze()
        if batch_average:
            return score.mean()
        else:
            return score
def prepare_image(image, resize=True):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)
def convert_to_y_channel(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:,:,0]
    return y
def compute_mse(image1, image2):
    img1 = np.array(image1)
    img2 = np.array(image2)
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    mse = np.mean((img1 - img2) ** 2)
    return mse
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
def compute_metrics(folder1_path, folder2_path, output_file='metrics_results.csv'):
    bar = progressbar.ProgressBar(maxval=len(os.listdir(folder1_path))).start()
    folder1_images = sorted(glob(os.path.join(folder1_path, '*')))
    folder2_images = sorted(glob(os.path.join(folder2_path, '*')))
    results = []
    i=0
    for img1_path, img2_path in zip(folder1_images, folder2_images):
        bar.update(i)
        img1_name = os.path.basename(img1_path)
        img2_name = os.path.basename(img2_path)
        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            img1_ten = lpips.im2tensor(lpips.load_image(img1_path)) # RGB image from [-1,1]
            img2_ten = lpips.im2tensor(lpips.load_image(img2_path))
            if dev=="cuda":
                img1_ten = img1_ten.cuda()
                img2_ten = img2_ten.cuda()
            ref  = prepare_image(Image.open(img1_path).convert("RGB"))
            dist = prepare_image(Image.open(img2_path).convert("RGB"))
            model = DISTS().to(torch.device(dev))
            ref = ref.to(torch.device(dev))
            dist = dist.to(torch.device(dev))
            y1 = convert_to_y_channel(img1)
            y2 = convert_to_y_channel(img2)
            psnr_value = calculate_psnr(y1, y2)
            ssim_value = ssim(y1, y2, data_range=255)
            mse_value = compute_mse(y1,y2)
            lpips_value = loss_fn.forward(img1_ten,img2_ten)
            dists_value = model(ref, dist)
            i=i+1
            results.append({
                'Image1': img1_name,
                'Image2': img2_name,
                'PSNR'  : psnr_value,
                'SSIM'  : ssim_value,
                'MSE'   : mse_value,
                'LPIPS' : lpips_value.item(),
                'DISTS' : dists_value.item()
            })
        except Exception as e:
            print(f"Error processing images {img1_name} and {img2_name}: {str(e)}")
    if results:
        df = pd.DataFrame(results)
        print("\nSummary Statistics:")
        print(f"Average PSNR: {df['PSNR'].mean():.3f}")
        print(f"Average SSIM: {df['SSIM'].mean():.3f}")
        print(f"Average MSE: {df['MSE'].mean():.3f}")
        print(f"Average LPIPS: {df['LPIPS'].mean():.3f}")
        print(f"Average DISTS: {df['DISTS'].mean():.3f}")
        df.to_csv(output_file, index=False)
        fields = ['', '', df['PSNR'].mean(),df['SSIM'].mean(),df['MSE'].mean(),df['LPIPS'].mean(),df['DISTS'].mean()]
        with open(output_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        print(f"\nResults saved to {output_file}")
GT = sys.argv[1]
RESULT = sys.argv[2]
RESULT_FILE = sys.argv[3]
dev = sys.argv[4]
loss_fn = lpips.LPIPS(net='alex', version='0.1')
if dev=="cuda":
	loss_fn.cuda()
result_path0=RESULT_FILE.split('/')[1]
os.makedirs(result_path0, exist_ok=True)
compute_metrics(GT, RESULT, output_file=RESULT_FILE)
stop = timeit.default_timer()
print('Tested images: ', str(len(os.listdir(RESULT))))
print('Inference time: ', stop - start, ' seconds')
if dev=="cpu":
    print('Running by CPU') 
else:
    print('Running by GPU') 