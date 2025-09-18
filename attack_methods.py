import torch
import numpy as np
import scipy.stats as st
import torch.nn.functional as F

"""Translation-Invariant https://arxiv.org/abs/1904.02884"""
#生成一个三通道的二维高斯核，用做图像处理中的卷积操作
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)   #生成以为高斯分布的坐标点
    kern1d = st.norm.pdf(x) #计算高斯分布的概率密度函数
    kernel_raw = np.outer(kern1d, kern1d)   #计算二维高斯核
    kernel = kernel_raw / kernel_raw.sum()  #对高斯核进行归一化处理
    kernel = kernel.astype(np.float32)  #将高斯核转换为float32类型
    #将二维高斯核扩展为三通道
    gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
    #增加一个维度，符合卷积核的格式
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
    #将高斯核转换为Pytorch张量并移动到GPU
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
    return gaussian_kernel

"""Input diversity: https://arxiv.org/abs/1803.06978"""
#对输入图像x进行多样化处理，通过随机缩放和填充操作生成不同版本的输入图像
#resize_rate表示缩放比例，#diversity_prob表示多样化概率
def DI(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0   #确保缩放比例大于等于1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0  #确保多样性概率在0到1之间
    img_size = x.shape[-1]  #获取原始图像的大小
    img_resize = int(img_size * resize_rate)    #计算缩放后的图像大小
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)    #随机生成一个新的图像大小，范围在原始大小和缩放大小之间
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)  #按照生成的大小对输入图像进行双线性插值缩放
    #计算需要填充的高度和宽度
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    #随机生成顶部和底部的填充值
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    #随机生成左侧和右侧的填充值
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    #对缩放后的图像进行填充，填充值为0
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    #以给定的多样性概率决定是否返回填充后的图像，否则返回原始图像
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret