Copyright (c) 2026 Author(s)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

# A Generalized Training Method for Dehazing Networks (FFA-Net Implementation)

This repository provides an implementation and training pipeline based on FFA-Net for image dehazing. It supports training and evaluation on RESIDE (ITS / OTS) and NH-HAZE4K datasets.
The project is designed to run directly after extracting the compressed archive. However, pretrained model files are also provided. If the required trained or pretrained models are missing, the program may raise a “model not found” error during execution.
See the section “Pretrained Models” below for details.

---

## 1. Requirements

This project is implemented using PyTorch.
The following external libraries are required:

- torch
- torchvision
- numpy
- opencv-python
- tqdm
- Pillow
- scikit-image

Additional packages may be needed depending on your environment. If an error occurs during execution, install the missing package (for example: pip install <package>).

---

## 2. Recommended environment

- Python 3.8 or higher
- CUDA-enabled GPU
- Linux or Windows

---

## 3. Dataset Structure

The code assumes the following directory layout:

FFA-Net
├── data
│   ├── RESIDE
│   │   ├── ITS
│   │   │   ├── train
│   │   │   │   ├── hazy
│   │   │   │   └── clear
│   │   │   └── test (SOTS – indoor)
│   │   │       ├── hazy
│   │   │       └── clear
│   │   ├── OTS
│   │   │   ├── train
│   │   │   │   ├── hazy
│   │   │   │   └── clear
│   │   │   └── test (SOTS – outdoor)
│   │   │       ├── hazy
│   │   │       └── clear
│   └── NH-HAZE4K (NH-HAZE, HAZE4K or mixed)
│       ├── train
│       │   ├── hazy
│       │   └── clear
│       └── test
│           ├── hazy
│           └── clear
└── net
     ├── test_imgs (hazy image)
     └── trained_models

The data files are not included in this repository.
If you wish to manually train the model with all or selected datasets, please download them from the links below:

RESIDE: https://sites.google.com/view/reside-dehaze-datasets/reside-standard
HAZE4K: https://github.com/liuye123321/DMT-Net
NH-HAZE: https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/

---

## 4. Training and Testing Datasets

A training and testing dataset can be selected using the command-line options --trainset and --testset.
All trained model files will be stored in the following directory: FFA-Net\net\trained_models

During training, models are automatically created and saved inside the directory. At least one trained model file in the directory will be used for testing.
If a model corresponding to the current dataset already exists, training automatically resumes from that model.
If no model is found, a new model is initialized and trained from scratch.
Models are saved periodically during training according to the eval_step setting.

Each checkpoint is stored using the following filename format:
its_train_ffa_3_19-step.pk

This pattern is used for each dataset, so multiple trained models can coexist in the same folder.

To run testing, simply keep the desired checkpoint file inside FFA-Net\net\trained_models.
The program automatically loads the correct model from there.

---

## 5. Training Commands

**ITS dataset:**
python main.py --net=ffa --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0001 --trainset=its_train --testset=its_test --steps=500000 --eval_step=50000

**OTS dataset:**
python main.py --net=ffa --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0001 --trainset=ots_train --testset=ots_test --steps=500000 --eval_step=50000

**NH-HAZE4K dataset:**
python main.py --net=ffa --crop --crop_size=240 --blocks=19 --gps=3 --bs=2 --lr=0.0001 --trainset=nh4k_train --testset=nh4k_test --steps=500000 --eval_step=50000

---

## 6. Testing Commands

**ITS:** python test.py --task=its --test_imgs=test_imgs
**OTS:** python test.py --task=ots --test_imgs=test_imgs
**NH-HAZE4K:** python test.py --task=nh4k --test_imgs=test_imgs
**Testing all models in test_imgs directory:** python test.py --test_imgs=test_imgs

During testing, the program automatically selects the correct model based on the filename prefix inside net/trained_models.
If the 'test_imgs' argument is not provided, the 'test_imgs' subdirectory will be used by default.

To ensure the model is detected correctly, the filenames must follow the rules below:

Dataset          | Required model prefix
-----------------|-------------------
RESIDE – ITS     | its_train_ffa_3_19
RESIDE – OTS     | ots_train_ffa_3_19
NH-HAZE4K        | nh4k_train_ffa_3_19

During training, checkpoints will be saved in the following format:
its_train_ffa_3_19-step.pk
ots_train_ffa_3_19-step.pk
nh4k_train_ffa_3_19-step.pk

During testing, the correct model(s) will be automatically loaded according to the option.

---

## 7. Pretrained Models & GUI App

We provide a total of seven pretrained FFA-Net models, trained under different dataset configurations to analyze the effect of dataset composition and augmentation on dehazing performance.
The pretrained models are placed in: FFA-Net/net/trained_models

**Models trained on original datasets**
1. ITS model: trained on the RESIDE ITS dataset (its_train_ffa_3_19.pk)
2. OTS model: trained on the RESIDE OTS dataset (ots_train_ffa_3_19.pk)
3. NH-HAZE model: trained on the original NH-HAZE dataset (nh_train_ffa_3_19.pk)
4. HAZE4K model: trained on the original HAZE4K dataset (haze4k_train_ffa_3_19.pk)

**Models trained with dataset augmentation and combination**
5. NH-HAZE (Augmented) (nh_train_ffa_3_19(aug).pk): trained on an augmented version of the NH-HAZE dataset to increase diversity in haze distribution and scene appearance.
6. NH-HAZE + HAZE4K (1:5) (nh4k_train_ffa_3_19(5by1).pk): trained on a combined dataset consisting of NH-HAZE and HAZE4K images mixed at a 1:5 ratio.
7. NH-HAZE + HAZE4K (1:10) (nh4k_train_ffa_3_19(10by1).pk): trained on a combined dataset consisting of NH-HAZE and HAZE4K images mixed at a 1:10 ratio.

This project also provides a GUI implemented using Tkinter.
Running the GUI (in the directory FFA-NET/net): python gui.py

At least one pretrained FFA-Net model should exist in FFA-Net/net/trained_models.

---

## 8. Citation & Reference

### Citation
@article{PLS26,
  title = {A Generalized Training Method for Dehazing Networks},
  author = {Park, Chaebin and Lee, Hyojae and Seo, Jieun and Lee, Sujin and Heo, Wonseok and Yoon, Maro and Nah, Jae-Ho},
  year = 2026,
  journal = {IPOL: Image Processing Online},
  note = {submitted.}
}

### Reference
Qin, Xu, et al. "FFA-Net: Feature fusion attention network for single image dehazing." Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 07. 2020.