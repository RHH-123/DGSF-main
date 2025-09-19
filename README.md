# DGSF
This repository contains the PyTorch code for the paper:
Dynamic Gradient Fusion Method with Local Spatial and Multi-Scale Frequency Transformations for Transferable Adversarial Attacks.
## Requirements
- python 3.8
- torch 1.8
- pretrainedmodels 0.7
- numpy 1.19
- pandas 1.2
## Qucik Start
### Prepare the data and models.
1. We have prepared the ImageNet-compatible dataset in this program and put the data in `./dataset/`.
2. The normally trained models (i.e., Inc-v3, Inc-v4, IncRes-v2, Res-50, Res-101, Res-100) are from "pretrainedmodels", if you use it for the first time, it will download the weight of the model automatically, just wait for it to finish.
3. The adversarially trained models (i.e, ens3_adv_inc_v3, ens4_adv_inc_v3, ens_adv_inc_res_v2) are from [SSA](https://github.com/yuyang-long/SSA) or [tf_to_torch_model](https://github.com/ylhz/tf_to_pytorch_model). For more detailed information on how to use them, visit these two repositories.
### DGSF Attack Method
All the provided codes generate adversarial examples on Inception_v3 model. If you want to attack other models, replace the model in `main()` function.
#### Runing attack
Using `dgsf_attack.py` to generate adversarial examples, you can run this attack as following
```
CUDA_VISIBLE_DEVICES=gpuid python dgsf_attack.py 
```
#### Evaluating the attack
The generated adversarial examples would be stored in directory `./outputs`. Then run the file `verify.py` to evaluate the success rate of each model used in the paper:
```
CUDA_VISIBLE_DEVICES=gpuid python verify.py
```
#### Evaluations on defenses
- [HGD](https://github.com/lfz/Guided-Denoise), [R&P](https://github.com/cihangxie/NIPS2017_adv_challenge_defense), [NIPS-r3](https://github.com/anlthms/nips-2017/tree/master/mmd): We directly run the code from the corresponding repo.
- [Bit-Red](https://github.com/thu-ml/ares/blob/main/ares/defense/bit_depth_reduction.py): step_num=4, alpha=200, base_model=Inc_v3_ens.
- [JPEG](https://github.com/thu-ml/ares/blob/main/ares/defense/jpeg_compression.py): No extra parameters.
- [FD](https://github.com/zihaoliu123/Feature-Distillation-DNN-Oriented-JPEG-Compression-Against-Adversarial-Examples): resize to 304\*304 for FD and then resize back to '299*299', base_model=Inc_v3_ens
- [ComDefend](https://github.com/jiaxiaojunQAQ/Comdefend): resize to 224\*224 for ComDefend and then resize back to 299\*299, base_model=Resnet_101
- [RS](https://github.com/locuslab/smoothing): noise=0.25, N=100, skip=100
- [NRP](https://github.com/Muzammal-Naseer/NRP): purifier=NRP, dynamic=True, base_model=Inc_v3_ens
