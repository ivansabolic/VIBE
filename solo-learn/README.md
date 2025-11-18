# Solo-Learn: Self-Supervised Learning with Backdoor Attacks

A focused implementation of self-supervised learning methods (All4One, BYOL, SimCLR) with integrated backdoor attack capabilities for research in robust representation learning.

**Built upon**: This codebase is adapted and extended from the excellent [solo-learn](https://github.com/vturrisi/solo-learn) library, which provides a comprehensive collection of self-supervised learning methods. 

## Overview

This repository provides implementations of popular self-supervised learning methods with built-in support for various backdoor attacks. The codebase is designed for researchers studying the vulnerability and robustness of self-supervised models against backdoor poisoning.

### Supported Methods
- **All4One**: Symbiotic Neighbour Contrastive Learning via Self-Attention and Redundancy Reduction
- **BYOL**: Bootstrap Your Own Latent
- **SimCLR**: Simple Framework for Contrastive Learning

### Supported Backdoor Attacks
- **BadNets**: Basic patch-based backdoor attacks
- **Blend**: Image blending with trigger patterns
- **WaNet**: Warping-based invisible backdoor attacks
- **FTrojan**: Fine-tuning based trojaning
- **Adaptive Blend**: Adaptive blending attacks
- **Adaptive Patch**: Adaptive patch-based attacks
- **Label Consistent (LC)**: Clean-label backdoor attacks

### Supported Datasets
- **CIFAR-10**: 32x32 natural images, 10 classes
- **CIFAR-100**: 32x32 natural images, 100 classes
- **ImageNet-30**: Subset of ImageNet with 30 classes

---

## Installation

Clone the repository and install the required dependencies:

For development installation:
```bash
pip install -e .
```

---

## Configuration Files

All configuration files follow the Hydra config system and are organized by dataset:

### CIFAR-10 Configs
Located in `scripts/pretrain/cifar/`:
- `all4one_badnets.yaml` - BadNets attack
- `all4one_blend.yaml` - Blend attack  
- `all4one_wanet.yaml` - WaNet attack
- `all4one_ftrojan.yaml` - FTrojan attack
- `all4one_adap_blend.yaml` - Adaptive Blend attack
- `all4one_adap_patch.yaml` - Adaptive Patch attack
- `all4one_lc.yaml` - Label Consistent attack

### CIFAR-100 Configs
Located in `scripts/pretrain/cifar100/`:
- `all4one_badnets.yaml`
- `all4one_blend.yaml`
- `all4one_wanet.yaml`
- `all4one_ftrojan.yaml`

### ImageNet-30 Configs
Located in `scripts/pretrain/imagenet30/`:
- `all4one_badnets.yaml`
- `all4one_blend.yaml`
- `all4one_wanet.yaml`
- `all4one_ftrojan.yaml`

---

## Running Experiments

### Basic Training

The general syntax for training is:

```bash
python main_pretrain.py \
    --config-path <path-to-config-folder> \
    --config-name <config-file-name> \
    [additional-arguments]
```

### Examples

**CIFAR-10 with different attacks:**
```bash
# BadNets attack
python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name all4one_badnets.yaml

# Blend attack
python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name all4one_blend.yaml

# WaNet attack
python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name all4one_wanet.yaml
```

**CIFAR-100 experiments:**
```bash
# BadNets on CIFAR-100
python main_pretrain.py --config-path scripts/pretrain/cifar100/ --config-name all4one_badnets.yaml

# Blend on CIFAR-100
python main_pretrain.py --config-path scripts/pretrain/cifar100/ --config-name all4one_blend.yaml
```

**ImageNet-30 experiments:**
```bash
# FTrojan on ImageNet-30
python main_pretrain.py --config-path scripts/pretrain/imagenet30/ --config-name all4one_ftrojan.yaml
```

### Customizing Training Parameters

You can override config parameters from the command line:

```bash
python main_pretrain.py \
    --config-path scripts/pretrain/cifar/ \
    --config-name all4one_badnets.yaml \
    ++max_epochs=500 \
    ++optimizer.batch_size=128 \
    ++backdoor.poisoning_rate=0.05
```

---

## Model Checkpoints

### Save Location
Trained models are automatically saved to the directory specified in the config files:
- **Default location**: `../solo_models/` (relative to the solo-learn directory)

**Note**: VIBE training uses these pretrained checkpoints as starting points. When using a newly trained checkpoint, ensure you update the corresponding model configuration in the VIBE training setup to point to the correct checkpoint path.

## Dataset Preparation

### CIFAR-10/CIFAR-100
These datasets are automatically downloaded when first used.

### ImageNet-30
Please take a look at VIBE README.

---
