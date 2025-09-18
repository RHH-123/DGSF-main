"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F #F提供一些常用的神经网络功能，比如激活函数、损失函数等
from torchvision.transforms import transforms as TF
from attack_methods import DI,gkern
from torchvision import transforms as T #导入torchvision库中的transforms模块，并将其别名为T,通常用于调用一些常用的图像转换函数
from tqdm import tqdm
import numpy as np
from PIL import Image
from dct import *
from cam import *
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
import random
from transform import *



parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='', help='Output directory with adversarial images.')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=4, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--num_copies", type=int, default=20, help="The number of Spectrum Transformations")
parser.add_argument("--rho", type=float, default=0.5, help="Tuning factor")
parser.add_argument("--sigma", type=float, default=16.0, help="Std of random noise")
parser.add_argument("--gamma", type=float, default=0.8, help="Std of random noise")


opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#定义图像的预处理操作，首先调整图像大小为299*299像素，然后将其转换为张量
transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)



#对输入张量进行裁剪，确保其值在t_min和t_max之间
def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

#保存对抗样本图像
def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)


T_kernel = gkern(7, 3)


ops_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
ops_bk_weights = [1, 1, 1, 1]
ops = [vertical_shift, horizontal_shift, vertical_flip, horizontal_flip, rotate180, random_scale, random_resize, add_uniform_noise, drop_out, Jitter]
ops_bk = [add_uniform_noise, add_salt_noise, add_gass_noise, add_poisson_noise]

def changeCAM(clist, move_max=15, resize_max=0.4):

    x_move_pix = random.randint(-move_max, move_max)
    y_move_pix = random.randint(-move_max, move_max)
    x1, y1, x2, y2 = clist
    resize_rate = random.uniform(-resize_max, resize_max)
    x_resize_pix = int((x1 - x2) * resize_rate)
    y_resize_pix = int((y1 - y2) * resize_rate)
    x1_ = min(max(x1 - x_resize_pix + x_move_pix, 13), 284)
    x2_ = max(min(x2 + x_resize_pix + x_move_pix, 284), 13)
    y1_ = min(max(y1 - y_resize_pix + y_move_pix, 13), 284)
    y2_ = max(min(y2 + y_resize_pix + y_move_pix, 284), 13)

    if x1_ == x2_:
        x2_ = x2_ + 13
        x1_ = x1_ - 13
    if y1 == y2:
        y2_ = y2_ + 13
        y1_ = y1_ - 13
    return x1_, y1_, x2_, y2_

def get_length(length):
    num_block = 3
    length = int(length)
    rand = np.random.uniform(size=num_block)
    rand_norm = np.round(rand * length / rand.sum()).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)


def shuffle(x):
    num_block = 3
    _, _, w, h = x.shape
    width_length, height_length = get_length(w), get_length(h)
    width_perm, height_perm = np.random.permutation(np.arange(num_block)), np.random.permutation(np.arange(num_block))


    x_split_w = torch.split(x, width_length, dim=2)
    x_w_perm = torch.cat([x_split_w[i] for i in width_perm], dim=2)  # 按照随机排列的顺序拼接


    x_split_h = torch.split(x_w_perm, height_length, dim=3)
    x_h_perm = torch.cat([x_split_h[i] for i in height_perm], dim=3)  # 按照随机排列的顺序拼接

    return x_h_perm


def truncated_normal(shape, mean=0.0, std=1.0, lower_bound=-2.0, upper_bound=2.0):

    x = torch.randn(shape) * std + mean


    while True:
        mask = (x < lower_bound) | (x > upper_bound)
        if not mask.any():
            break
        x[mask] = torch.randn(mask.sum()) * std + mean  # 重新生成超出范围的元素
    return x


def shuffle_rotate(x):
    num_block = 3
    B, C, w, h = x.shape
    width_length, height_length = get_length(w), get_length(h)
    width_perm, height_perm = np.random.permutation(np.arange(num_block)), np.random.permutation(np.arange(num_block))

    # 按照 width_length 切分
    x_split_w = torch.split(x, width_length, dim=2)

    x_w_perm = torch.cat([x_split_w[i] for i in width_perm], dim=2)  # 打乱宽度方向

    # 按照 height_length 切分
    x_spilt_h_l = [torch.split(x_split_w[i], height_length, dim=3) for i in width_perm]

    # 对每个 strip 进行旋转
    x_h_perm = []
    for strip in x_spilt_h_l:
        strip_perm = []
        for i in range(len(strip)):


            # 检查 strip 的高度和宽度是否为零，如果是则跳过旋转
            if strip[i].shape[2] == 0 or strip[i].shape[3] == 0:

                continue


            angle = torch.randn(1).item() * 5.0
            rotated_strip = TF.rotate(strip[i], angle=angle, interpolation=TF.InterpolationMode.BILINEAR)
            strip_perm.append(rotated_strip)


        if strip_perm:
            x_h_perm.append(torch.cat(strip_perm, dim=3))


    if x_h_perm:
        return torch.cat(x_h_perm, dim=2)
    else:
        print("Error: All strips have invalid dimensions after rotation.")
        return x

def blocktransform(x, cam_list, choice=-1):

    x_copy = x.clone()

    for bat, _ in enumerate(x_copy):
        x1, y1, x2, y2 = changeCAM(cam_list[bat])

        def select_operation(ops_list, weights, override_choice):
            """
            根据权重选择操作
            :param ops_list: 操作列表
            :param weights: 权重列表
            :param override_choice: 强制指定操作索引（-1 表示随机选择）
            """
            if override_choice >= 0:
                return override_choice
            return np.random.choice(len(ops_list), p=np.array(weights) / sum(weights))
        # 对每个小块选择操作并应用
        chosen = select_operation(ops_bk, ops_bk_weights, choice)
        x_copy[bat, :, 0:x1, 0:y1] = ops_bk[chosen](
            x_copy[bat, :, 0:x1, 0:y1].unsqueeze(0)
        )

        chosen = select_operation(ops_bk, ops_bk_weights, choice)
        x_copy[bat, :, x1:x2, 0:y1] = ops_bk[chosen](
            x_copy[bat, :, x1:x2, 0:y1].unsqueeze(0)
        )

        chosen = select_operation(ops_bk, ops_bk_weights, choice)
        x_copy[bat, :, x2:299, 0:y1] = ops_bk[chosen](
            x_copy[bat, :, x2:299, 0:y1].unsqueeze(0)
        )

        chosen = select_operation(ops_bk, ops_bk_weights, choice)
        x_copy[bat, :, 0:x1, y1:y2] = ops_bk[chosen](
            x_copy[bat, :, 0:x1, y1:y2].unsqueeze(0)
        )

        chosen = select_operation(ops, ops_weights, choice)
        x_copy[bat, :, x1:x2, y1:y2] = ops[chosen](
            x_copy[bat, :, x1:x2, y1:y2].unsqueeze(0)
        )  # 显著区域用 ops 的变换

        chosen = select_operation(ops_bk, ops_bk_weights, choice)
        x_copy[bat, :, x2:299, y1:y2] = ops_bk[chosen](
            x_copy[bat, :, x2:299, y1:y2].unsqueeze(0)
        )

        chosen = select_operation(ops_bk, ops_bk_weights, choice)
        x_copy[bat, :, 0:x1, y2:299] = ops_bk[chosen](
            x_copy[bat, :, 0:x1, y2:299].unsqueeze(0)
        )

        chosen = select_operation(ops_bk, ops_bk_weights, choice)
        x_copy[bat, :, x1:x2, y2:299] = ops_bk[chosen](
            x_copy[bat, :, x1:x2, y2:299].unsqueeze(0)
        )

        chosen = select_operation(ops_bk, ops_bk_weights, choice)
        x_copy[bat, :, x2:299, y2:299] = ops_bk[chosen](
            x_copy[bat, :, x2:299, y2:299].unsqueeze(0)
        )


    return shuffle_rotate(x_copy)   #返回经过变换后的图像

def freq(x, image_width, sigma, rho):
    x_copy = x.clone()
    gauss = torch.randn(x_copy.size()) * (sigma / 255)  # 生成标准差为sigma的随机高斯噪声
    gauss = gauss.cuda()
    x_dct = dct_2d(x_copy + gauss).cuda()
    mask = (torch.rand_like(x_copy) * 2 * rho + 1 - rho).cuda()
    x_idct = idct_2d(x_dct * mask)
    return x_idct

def spatial_transform(x, cam_list, num_copies):
    """
    对输入进行缩放以用于块洗牌（BlockShuffle）
    """
    # 通过多次调用blocktransform生成多个副本并连接它们
    return torch.cat(
        [blocktransform(x, cam_list) for _ in range(num_copies)]
    )

def freq_transform(x, num_copies, image_width, sigma, rho):
    """
    对输入进行缩放以用于块洗牌（BlockShuffle）
    """
    # 通过多次调用blocktransform生成多个副本并连接它们
    return torch.cat(
        [freq(x, image_width, sigma, rho) for _ in range(num_copies)]
    )

def block_trans(image, block):
    chosen_op = ops[np.random.randint(0, high=len(ops), dtype=np.int32)]
    x_start, x_end, y_start, y_end = block
    x_block = image[:, x_start:x_end, y_start:y_end]
    return chosen_op(x_block.unsqueeze(0)).squeeze(0)

def SFMA_FGSM_local(images, gt, model, min_val, max_val):
    image_width = opt.image_width
    momentum = opt.momentum
    num_iter = 10
    eps = opt.max_epsilon / 255.0
    alpha = eps / num_iter
    images_adv = images.clone()
    grad = 0
    rho = opt.rho
    num_copies = opt.num_copies
    sigma = opt.sigma
    gamma = opt.gamma


    scales_and_weights = [(1.0, 0.7), (0.75, 0.2), (0.5, 0.1)]


    gt_expanded = gt.repeat(num_copies)

    cam_list = []
    for image in images:
        image = image.cuda()

        inception_model = pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda()


        model = torch.nn.Sequential(Normalize(opt.mean, opt.std), inception_model)


        cam = CAM(image, inception_model, 0.5)
        cam_list.append(cam)

    for t in range(num_iter):
        x_adv = images_adv.clone()
        x_adv = V(x_adv, requires_grad=True)
        grad_spatial = 0
        grad_freq = 0
        loss_spatial_total = 0
        loss_freq_total = 0
        #空间域处理
        x_spatial = spatial_transform(x_adv,cam_list,num_copies)
        output_spatial = model(x_spatial)
        loss_spatial = F.cross_entropy(output_spatial, gt_expanded)
        grad_spatial = torch.autograd.grad(
            loss_spatial, x_adv, retain_graph=False, create_graph=False
        )[0]
        loss_spatial_avg = loss_spatial / num_copies

        #频域处理

        for scale, weight in scales_and_weights:
            scaled_x = F.interpolate(x_adv, scale_factor=scale, mode='bilinear', align_corners=False)
            scaled_x = V(scaled_x, requires_grad=True)  #
            scale_grad = 0
            scale_loss = 0
            # scale_freq_images = []
            x_freq = freq_transform(scaled_x, num_copies, image_width, sigma, rho)
            output_freq = model(x_freq)
            loss_freq = F.cross_entropy(output_freq, gt_expanded)
            scale_grad += torch.autograd.grad(
                loss_freq, scaled_x, retain_graph=False, create_graph=False
            )[0]
            # 取该尺度下的平均噪声并加权
            scale_grad_resized = F.interpolate(scale_grad, size=(image_width, image_width), mode='bilinear', align_corners=False)
            grad_freq += scale_grad_resized * weight
            loss_freq_total += weight * (loss_freq / num_copies)
            #loss_freq_total += weight * loss_freq

        grad_freq /= sum([w for _, w in scales_and_weights])
        loss_freq_avg = loss_freq_total


        # 反转损失对权重的影响，损失越大，权重越大
        weight_spatial = np.exp(gamma * (loss_freq_avg.cpu().item() - loss_spatial_avg.cpu().item())) / (
                np.exp(gamma * (loss_spatial_avg.cpu().item() - loss_freq_avg.cpu().item())) +
                np.exp(gamma * (loss_freq_avg.cpu().item() - loss_spatial_avg.cpu().item()))
        )

        weight_freq = np.exp(gamma * (loss_spatial_avg.cpu().item() - loss_freq_avg.cpu().item())) / (
                np.exp(gamma * (loss_spatial_avg.cpu().item() - loss_freq_avg.cpu().item())) +
                np.exp(gamma * (loss_freq_avg.cpu().item() - loss_spatial_avg.cpu().item()))
        )

        # 动态加权后的梯度融合
        grad_combined = weight_spatial * grad_spatial + weight_freq * grad_freq
        #grad_combined = F.conv2d(grad_combined, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        grad_combined = grad_combined / torch.abs(grad_combined).mean([1, 2, 3], keepdim=True)  # 归一化噪声
        grad_combined = momentum * grad + grad_combined  # 使用动量更新噪声
        grad = grad_combined  # 保存当前的梯度

        # 按照FGSM方法更新对抗样本
        images_adv = images_adv + alpha * torch.sign(grad_combined)
        images_adv = clip_by_tensor(images_adv, min_val, max_val)  # 保证像素值在[min, max]范围内

    return images_adv.detach()  # 返回生成的对抗样本


def main():

    model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())


    X = ImageNet(opt.input_dir, opt.input_csv, transforms)

    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)


    for images, images_ID,  gt_cpu in tqdm(data_loader):


        gt = gt_cpu.cuda()

        images = images.cuda()

        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)

        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)

        adv_img =SFMA_FGSM_local(images, gt, model, images_min, images_max)

        adv_img_np = adv_img.cpu().numpy()

        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255

        save_image(adv_img_np, images_ID, opt.output_dir)


if __name__ == '__main__':
    main()