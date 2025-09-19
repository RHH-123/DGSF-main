import os
import torch
from torch.autograd import Variable as V
from torch import nn

from torchvision import transforms as T
from Normalize import Normalize, TfNormalize
from loader import ImageNet
from torch.utils.data import DataLoader

from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
)


import torchvision.models as models

torch.cuda.empty_cache()

batch_size = 10

input_csv = './dataset/images.csv'
input_dir = './dataset/images'
adv_dir = ''
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def get_model(net_name, model_dir):
    """Load converted model"""

    if net_name == 'pytorch_vgg16':

        vgg16 = models.vgg16(pretrained=True)

        model = nn.Sequential(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vgg16.eval().cuda(),
        )
        return model

    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf2torch_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf2torch_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf2torch_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf2torch_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf2torch_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf2torch_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf2torch_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        raise ValueError('Wrong model name!')


    model = nn.Sequential(

        TfNormalize('tensorflow'),
        net.KitModel(model_path).eval().cuda(),
    )
    return model


def verify(model_name, path):
    model = get_model(model_name, path)


    if model_name == 'pytorch_vgg16':
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
    else:
        transform = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor()
        ])


    X = ImageNet(adv_dir, input_csv, transform)
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True,
                             num_workers=8)
    sum = 0


    is_pytorch_vgg16 = model_name == 'pytorch_vgg16'

    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            if is_pytorch_vgg16:
                outputs = model(images)
                sum += (outputs.argmax(1) != gt).detach().sum().cpu()
            else:
                sum += (model(images)[0].argmax(1) != (gt + 1)).detach().sum().cpu()


    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))



def main():

    model_names = [
        'tf2torch_inception_v3',
        'tf2torch_inception_v4',
        'tf2torch_inc_res_v2',
        'tf2torch_resnet_v2_50',
        'tf2torch_resnet_v2_101',
        'tf2torch_resnet_v2_152',
        'tf2torch_ens3_adv_inc_v3',
        'tf2torch_ens4_adv_inc_v3',
        'tf2torch_ens_adv_inc_res_v2',
        'pytorch_vgg16'
    ]

    models_path = './models/'

    for model_name in model_names:
        verify(model_name, models_path)
        print("===================================================")


if __name__ == '__main__':
    main()