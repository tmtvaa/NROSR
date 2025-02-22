import warnings
warnings.filterwarnings('ignore')
import sys
import os
import glob
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
from collections import OrderedDict
from natsort import natsorted
import progressbar
import timeit
start = timeit.default_timer()
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer
def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer
def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer
def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding
class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule
    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output
    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule
    def forward(self, x):
        output = x + self.sub(x)
        return output
    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr
def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)
def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)    
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device('cuda'))
    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 
class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''
    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC': 
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res
class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''
    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA', noise_input=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.noise = GaussianNoise() if noise_input else None
        self.conv1x1 = conv1x1(nc, gc)
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x2 = x2 + self.conv1x1(x)
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return self.noise(x5.mul(0.2) + x)
class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''
    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.noise = GaussianNoise()
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return self.noise(out.mul(0.2) + x)
def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)
def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)
class RRDB_Net_hm(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4,
                 norm_type=None, act_type='leakyrelu',
                 mode='CNA', upsample_mode='upconv'):
        super(RRDB_Net_hm, self).__init__()
        # Calculate number of upscale layers
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        # 1. Feature Extraction Layer
        self.feature_extraction = nn.Sequential(
            conv_block(in_nc, nf, kernel_size=3,
                         norm_type=None, act_type=None)
        )
        # 2. Residual in Residual Dense Blocks (RRDB)
        self.residual_blocks = nn.ModuleList([
            RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True,
                   pad_type='zero', norm_type=norm_type,
                   act_type=act_type, mode='CNA')
            for _ in range(nb)
        ])
        # 3. Low-Resolution Convolution
        self.lr_conv = conv_block(nf, nf, kernel_size=3,
                                    norm_type=norm_type,
                                    act_type=None, mode=mode)
        # 4. Upsampling Module
        self.upsampler = self._create_upsampler(
            nf, upscale, upsample_mode, act_type
        )
        # 5. High-Resolution Convolution Layers
        self.hr_conv0 = conv_block(nf, nf, kernel_size=3,
                                     norm_type=None,
                                     act_type=act_type)
        self.hr_conv1 = conv_block(nf, out_nc, kernel_size=3,
                                     norm_type=None,
                                     act_type=None)
        self.hr_conv2 = conv_block(nf, nf, kernel_size=3,
                                     norm_type=None,
                                     act_type=act_type)
        self.hr_conv3 = conv_block(nf, out_nc, kernel_size=3,
                                     norm_type=None,
                                     act_type=None)
    def _create_upsampler(self, nf, upscale, upsample_mode, act_type):
        if upsample_mode == 'upconv':
            upsample_block =  upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = pixelshuffle_block
        else:
            raise NotImplementedError(
                f'Upsample mode [{upsample_mode}] is not found'
            )
        if upscale == 3:
            return nn.Sequential(
                upsample_block(nf, nf, 3, act_type=act_type)
            )
        else:
            return nn.Sequential(*[
                upsample_block(nf, nf, act_type=act_type)
                for _ in range(int(math.log(upscale, 2)))
            ])
    def forward(self, x):
        # 1. Feature Extraction
        fea = self.feature_extraction(x)
        # 2. Residual Blocks
        residual = fea
        for block in self.residual_blocks:
            fea = block(fea)
        # 3. Low-Resolution Convolution
        fea = self.lr_conv(fea)
        # 4. Residual Connection
        fea = fea + residual
        # 5. Upsampling
        fea = self.upsampler(fea)
        # 6. High-Resolution Convolutions
        fea1 = self.hr_conv0(fea)
        out1 = self.hr_conv1(fea1)
        fea2 = self.hr_conv2(fea)
        out2= self.hr_conv3(fea2)
        return out1,out2
def adaptive_watershed_transform(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel_size = int(np.mean(gray.shape) / 94) * 25 + 8 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    sure_bg = cv2.dilate(adaptive_thresh, kernel, iterations=1)
    dist_transform = cv2.distanceTransform(adaptive_thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0 
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  
    return image,markers
def adaptive_watershed_transform_from_img(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7),1)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel_size = int(np.mean(gray.shape) / 94) * 25 +8 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    sure_bg = cv2.dilate(adaptive_thresh, kernel, iterations=1)
    dist_transform = cv2.distanceTransform(adaptive_thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0  
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  
    return image,markers
model_path = sys.argv[1]
test_img_folder_path = sys.argv[2]
result_outfolder = sys.argv[3]
dev = sys.argv[4]
device = torch.device(dev)
test_img_folder = natsorted(sorted(glob.glob(test_img_folder_path + "/*.*"), key=len))
model_name = model_path.split('/')[-1].split(".")[0]
print("Provided Model: ", model_name)
result_path0 = result_outfolder
os.makedirs(result_path0, exist_ok=True)
result_path = result_path0 +'/pred_hr/'
os.makedirs(result_path, exist_ok=True)
result_path2 = result_path0 + '/pred_hr_heatmap/'
os.makedirs(result_path2, exist_ok=True)
result_path3 = result_path0 +'/pred_hr_enh/'
os.makedirs(result_path3, exist_ok=True)
result_path4= result_path0 + '/pred_hr_enh1/'
os.makedirs(result_path4, exist_ok=True)
result_path5= result_path0 + '/pred_hr_enh2/'
os.makedirs(result_path5, exist_ok=True)
result_path6= result_path0 + '/pred_hr_enh3/'
os.makedirs(result_path6, exist_ok=True)
model = RRDB_Net_hm(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                         mode='CNA', upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=False)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
bar = progressbar.ProgressBar(maxval=len(os.listdir(test_img_folder_path))).start()
model = model.to(device)
print('Model path {:s}. \nTesting...'.format(model_path))
for idx, path in enumerate(test_img_folder):
    bar.update(idx)
    base = os.path.splitext(os.path.basename(path))[0].split(".jpg")[0]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    output1 = model(img_LR)[0].data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output1 = np.transpose(output1[[2, 1, 0], :, :], (1, 2, 0))
    output1 = (output1 * 255.0).round()
    cv2.imwrite(result_path + '{:s}.png'.format(base), output1)
    output2 = model(img_LR)[1].data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output2 = np.transpose(output2[[2, 1, 0], :, :], (1, 2, 0))
    output2 = (output2 * 255.0).round()
    cv2.imwrite(result_path2 + '{:s}.png'.format(base), output2)
for name in os.listdir(result_path2):
    enhanced_image0 = cv2.imread(result_path+name)
    enhanced_image1 = cv2.fastNlMeansDenoisingColored(enhanced_image0, None, 1, 1, 1, 21)
    enhanced_image2 = cv2.fastNlMeansDenoisingColored(enhanced_image0, None, 2, 2, 3, 21)
    enhanced_image3 = cv2.fastNlMeansDenoisingColored(enhanced_image0, None, 3, 3, 5, 21)
    cv2.imwrite(result_path4 + name, enhanced_image1)
    cv2.imwrite(result_path5 + name, enhanced_image2)
    cv2.imwrite(result_path6 + name, enhanced_image3)
    result_image1 , markers1= adaptive_watershed_transform(result_path2+name)
    result_image2 , markers2= adaptive_watershed_transform(result_path+name)
    result_image3 , markers3= adaptive_watershed_transform_from_img(enhanced_image1)
    result_image4 , markers4= adaptive_watershed_transform_from_img(enhanced_image2)
    result_image5 , markers5= adaptive_watershed_transform_from_img(enhanced_image3)
    y1 = np.clip(markers1, 0, 255).astype(np.uint8)
    y2 = np.clip(markers2, 0, 255).astype(np.uint8)
    y3 = np.clip(markers3, 0, 255).astype(np.uint8)
    y4 = np.clip(markers4, 0, 255).astype(np.uint8)
    y5 = np.clip(markers5, 0, 255).astype(np.uint8)
    _, binary_image1 = cv2.threshold(y1, 1, 255, cv2.THRESH_BINARY)
    _, binary_image2 = cv2.threshold(y2, 1, 255, cv2.THRESH_BINARY)
    _, binary_image3 = cv2.threshold(y3, 1, 255, cv2.THRESH_BINARY)
    _, binary_image4 = cv2.threshold(y4, 1, 255, cv2.THRESH_BINARY)
    _, binary_image5 = cv2.threshold(y5, 1, 255, cv2.THRESH_BINARY)
    score1= abs(np.sum(1-binary_image1) - np.sum(1-binary_image2))
    score2= abs(np.sum(1-binary_image1) - np.sum(1-binary_image3))
    score3= abs(np.sum(1-binary_image1) - np.sum(1-binary_image4))
    score4= abs(np.sum(1-binary_image1) - np.sum(1-binary_image5))
    scores = [score1,score2,score3,score4]
    min_value = min(scores)
    min_index = scores.index(min_value)
    if min_index ==0:
        cv2.imwrite(result_path3+name, enhanced_image0)
    elif min_index ==1:
        cv2.imwrite(result_path3+name, enhanced_image1)
    elif min_index ==2:
        cv2.imwrite(result_path3+name, enhanced_image2)
    else:
        cv2.imwrite(result_path3+name, enhanced_image3)
stop = timeit.default_timer()
print('Tested images: ', str(len(os.listdir(test_img_folder_path))))
print('Inference time: ', stop - start, ' seconds')
if dev=="cpu":
    print('Running by CPU') 
else:
    print('Running by GPU') 
