

# UniCross
UniCross: Balanced Multimodal Learning for Alzheimer’s Disease Diagnosis by Uni-modal Separation and Metadata-guided Cross-modal Interaction

<img src="Image/fig1.png" alt="Screenshot" width="60%">

## Content 

- [Introduction](#Introduction)
- [File Tree](#File Tree)
- [Train steps](#Train steps)
- [License](#License)
- [Thanks](#Thanks)

### Introduction

This repo is for the MICCAI 2025 accepted paper UniCross: [link](https://papers.miccai.org/miccai-2025/0972-Paper2409.html)

### File Tree 

```
UniCross
├── Data_process
│   ├── Clinical_Data_pre_processed.py
│   └── Image_Data_pre_processed.py
├── LICENSE
├── README.md
├── linear_prob.py
├── loss
│   ├── MetaWeightContrastiveLoss.py
│   └── loss.py
├── model
│   ├── certainty_aware_fusion_module.py
│   ├── fusion_modules.py
│   ├── mynet.py
│   ├── uncertainty_fusion_module.py
│   └── vit3d.py
├── mydataset.py
├── train_joint.py
├── train_stage1.py 
├── train_stage2.py
├── train_unimodal.py
└── utils
    ├── __init__.py
    └── utils.py

```

### Train steps
1. Download Dataset (FDG-PET and sMRI and ADNIMERGE table) from [ADNI](https://adni.loni.usc.edu/)
2. Data Pre-Processing
3. Train Stage1 and Train Stage2


### License

Distributed under the MIT License. See [`LICENSE`](https://github.com/shaojintian/Best_README_template/blob/master/LICENSE.txt)for more information.

### Thanks

- [ViT_recipe_for_AD](https://github.com/qasymjomart/ViT_recipe_for_AD)
- [DI-MML](https://github.com/fanyunfeng-bit/DI-MML)
- [SupConLoss](https://github.com/XG293/SupConLoss)




