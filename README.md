# Attention-based Batch Normalization for Binary Neural Networks (ABN-BNN)

This repository contains the official PyTorch implementation of our paper:

**"Attention-based Batch Normalization for Binary Neural Networks"**

##  Environment Setup

We recommend using the following Python environment:

- Python ≥ 3.8
- PyTorch ≥ 1.10.0
- torchvision ≥ 0.11.0
- CUDA ≥ 11.1 (if using GPU)
- numpy
- matplotlib
- tqdm
- Pillow
- `pytorch-grad-cam` (for visualization, optional)

## Reproducing Our Results

To reproduce the experiments in our paper, run the following commands:

```bash
# ResNet with standard BN
python main_binary.py --model resnet_binary --save resnet_cifar100 --dataset cifar100 --input_size 32 --epochs 200 -b 256

# ResNet with ABN (Attention-based Batch Normalization)
python main_binary.py --model resnet_abn_binary --save resnet_cifar100_ABN --dataset cifar100 --input_size 32 --epochs 200 -b 256

# VGG with standard BN
python main_binary.py --model vgg_cifar100_binary --save vgg_cifar100 --dataset cifar100 --input_size 32 --epochs 200 -b 256

# VGG with ABN
python main_binary.py --model vgg_abn_cifar100_binary --save vgg_cifar100_ABN --dataset cifar100 --input_size 32 --epochs 200 -b 256
```


