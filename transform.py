"""Implementation of sample attack."""

import torch.nn.functional as F
from torchvision import transforms as T
from cam import *
import random


# 定义各种空间变换操作
#对x沿垂直方向（宽度维度）进行随机平移
def vertical_shift(x):
    _, _, w, _ = x.shape
    step = np.random.randint(low=0, high=w, dtype=np.int32)
    return x.roll(step, dims=2)

#对x沿水平方向（高度维度）进行随机平移
def horizontal_shift(x):
    _, _, _, h = x.shape
    step = np.random.randint(low=0, high=h, dtype=np.int32)
    return x.roll(step, dims=3)

#对输入x沿垂直方向进行翻转
def vertical_flip(x):
    return x.flip(dims=(2,))

#对输入x沿水平方向进行翻转
def horizontal_flip(x):
    return x.flip(dims=(3,))

#将x旋转180度
def rotate180(x):
    return x.rot90(k=2, dims=(2,3))


def random_rotate(x):

    angle = random.uniform(0, 360)
    rotate_transform = T.RandomRotation(degrees=(angle, angle))
    return rotate_transform(x)

#对x进行随机缩放，通常用于数据增强
def random_scale(x):
    return torch.rand(1)[0] * x


def random_resize(x):
    _, _, w, h = x.shape
    scale_factor = random.uniform(0.6, 0.9)
    # scale_factor = 0.8
    new_h = int(h * scale_factor) + 1
    new_w = int(w * scale_factor) + 1

    x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 1)
    return x


def add_uniform_noise(x):
    return torch.clip(x + torch.zeros_like(x).uniform_(-16/255, 16/255), 0, 1)

def add_gass_noise(x):
    gauss =  torch.randn(x.size()) * (16.0 / 255)  # 生成标准差为sigma的随机高斯噪声
    gauss = gauss.cuda()
    return torch.clip(x + gauss, 0, 1)

def add_salt_noise(x,prob=0.01):
    noise = torch.rand_like(x)  # 生成 [0, 1] 的均匀分布随机数
    salt = (noise < prob / 2).float()  # 盐噪声（设置为1）
    pepper = (noise > 1 - prob / 2).float()  # 椒噪声（设置为0）
    noisy_x = x * (1 - salt - pepper) + salt  # 应用椒盐噪声
    return noisy_x

def add_poisson_noise(x, scale=30):
    """
    向输入张量添加泊松噪声
    :param x: 输入张量，范围应为 [0, 1]
    :param scale: 缩放因子，用于调整噪声强度
    :return: 添加噪声后的张量
    """
    device = x.device  # 确保在同一设备上
    x_scaled = x * scale  # 将图像值放大，模拟光子计数
    poisson_noise = torch.poisson(x_scaled).to(device)  # 生成泊松噪声
    noisy_x = poisson_noise / scale  # 缩放回 [0, 1] 范围
    return torch.clip(noisy_x, 0, 1)


#对x进行随机丢弃部分像素
def drop_out(x):
    return F.dropout2d(x, p=0.1, training=True) #对x的通道进行二维随机失活，失活的概率为10%，相当于随机将部分像素设为0

Blur = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
solarizer = T.RandomSolarize(threshold=0.6)


def Solarize(x):

    solarized_imgs = [solarizer(x) for _ in range(4)]
    x = solarized_imgs[0] + x
    return x


def AdjustSharpness(x):
    sharpness = transforms.RandomAdjustSharpness(sharpness_factor=10, p=0.5)
    x = sharpness(x)
    return x


def Invert(x):
    invert = T.RandomInvert(p=1)

    x = invert(x)
    return x


def Jitter(x):
    jitter = T.ColorJitter(brightness=1.5, contrast=3)
    x = jitter(x)
    return x