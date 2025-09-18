# DGSF
This repository contains the PyTorch code for the paper:
Dynamic Gradient Fusion Method with Local Spatial and Multi-Scale Frequency Transformations for Transferable Adversarial Attacks
## Requirements
- python 3.8
- torch 1.8
- pretrainedmodels 0.7
- numpy 1.19
- pandas 1.2
## Qucik Start
### Prepare models
Download pretrained PyTorch models [here](https://github.com/ylhz/tf_to_pytorch_model), which are converted from widely used Tensorflow models. Then put these models into './models/'
### DGSF Attack Method
### Evaluations on defenses
- [HGD](https://github.com/lfz/Guided-Denoise), [R&P](https://github.com/cihangxie/NIPS2017_adv_challenge_defense), [NIPS-r3](https://github.com/anlthms/nips-2017/tree/master/mmd): We directly run the code from the corresponding repo.
- [Bit-Red](https://github.com/thu-ml/ares/blob/main/ares/defense/bit_depth_reduction.py): step_num=4, alpha=200, base_model=Inc_v3_ens.
- [JPEG](https://github.com/thu-ml/ares/blob/main/ares/defense/jpeg_compression.py): No extra parameters.
- [FD](https://github.com/zihaoliu123/Feature-Distillation-DNN-Oriented-JPEG-Compression-Against-Adversarial-Examples): resize to 304\*304 for FD and then resize back to '299*299', base_model=Inc_v3_ens
- [ComDefend](https://github.com/jiaxiaojunQAQ/Comdefend): resize to 224\*224 for ComDefend and then resize back to 299\*299, base_model=Resnet_101
- [RS](https://github.com/locuslab/smoothing): noise=0.25, N=100, skip=100
- [NRP](https://github.com/Muzammal-Naseer/NRP): purifier=NRP, dynamic=True, base_model=Inc_v3_ens
