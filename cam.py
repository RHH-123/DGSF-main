import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import nn
from torchvision import transforms


class Flatten(nn.Module):
    """One layer module that flattens its input."""

    def __init__(self):
        super(Flatten, self).__init__() #初始化父类

    def forward(self, x):
        return x.view(x.size(0), -1)    #将输入张量展平为二维，形状为(batchisize,-1)



def GradCAM(img, c, features_fn, classifier_fn):
    torch.set_grad_enabled(True)
    feats = features_fn(img)
    _, N, H, W = feats.size()
    out = classifier_fn(feats)
    c_score = out[0, c]
    grads = torch.autograd.grad(c_score, feats)
    w = grads[0][0].mean(-1).mean(-1)
    sal = torch.matmul(w, feats.view(N, H * W))
    sal = sal.view(H, W).cpu().detach().numpy()
    sal = np.maximum(sal, 0)
    return sal


data_transform = transforms.Compose(
    [transforms.Resize([299, 299]), transforms.ToTensor()]
)


def CAM(image, model, th):
    # Split model in two parts
    arch = model.__class__.__name__
    # features_fn 是 model.features，直接调用特征提取方法
    features_fn = model.features
    # classifier_fn 是 model.logits，直接调用分类器部分
    classifier_fn = model.logits

    model.eval()
    with torch.no_grad():
        c = torch.argmax(model(image.unsqueeze(0)), dim=1).squeeze(0)
        sal = GradCAM(image.unsqueeze(0), int(c), features_fn, classifier_fn)
        sal = Image.fromarray(sal)
        sal = sal.resize((299, 299))
        tensor = torch.from_numpy(np.array(sal))


        th = (tensor.max() - tensor.min()) * th + tensor.min()


        x1 = (tensor >= th).sum(1).nonzero()[0].item()
        x2 = (tensor >= th).sum(1).nonzero()[-1].item()
        y1 = (tensor >= th).sum(0).nonzero()[0].item()
        y2 = (tensor >= th).sum(0).nonzero()[-1].item()

        x1 = min(max(x1, 27), 232)
        x2 = max(min(x2, 285), 67)
        y1 = min(max(y1, 27), 232)
        y2 = max(min(y2, 285), 67)

        if x1 == x2:
            x2 = x2 + 13
            x1 = x1 - 13
        if y1 == y2:
            y2 = y2 + 13
            y1 = y1 - 13

    return x1, y1, x2, y2
