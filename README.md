# SSD implementation from Luffic with AdaIN option

This repository implements [SSD (Single Shot MultiBox Detector)](https://github.com/lufficc/SSD). The code is modified to have a new parse argument so as to be able to use an [AdaIN](https://github.com/irasin/Pytorch_AdaIN) model in real time (at batch level) to modify the style of the source images this increasing variability of the source dataset and robustness to cross domain operation. [Paper](https://drive.google.com/file/d/1bSLNHFaw-HXJPNstRJRNo97sxEU9WyCj/view?usp=sharing)

## Main modifications to original code

### SSD/

#### *Modified*

train.py


### SSD/ssd/

#### *Added*

adain/

engine/styletransfer.py

datasets/clip.py


#### *Modified*

engine/trainer.py

config/path_catlog.py

data/datasets/voc.py

data/datasets/init.py

data/build.py

## Installation

```bash
git clone https://github.com/Federico-Maresca/Domain_Adaptation_Project.git
cd SSD
# Required packages: torch torchvision yacs tqdm opencv-python vizer
pip install -r requirements.txt

# Unzip adain
unzip ./ssd/adain/pretrained_adain.zip

# It's recommended to install the latest release of torch and torchvision.
```


## Train

### Setting Up Datasets
#### Pascal VOC

Folder structure must be as below
```
WORKING_DIRECTORY
 |_SSD
 |_dataset
     |_VOC2007
     |_VOC2012
     |_clipart
     |_dt_clipart
        |_VOC2007
        |_VOC2007
```

The Pascal VOC datasets are kept as the original folders from the PascalVOC website.

## TRAINING
```bash
# for example, train SSD300:
python train.py --config-file configs/your_config_file_here.yaml --ckpt ./outputs/DA_project_baseline/model_final.pth
```
### Training with AdaIN
A different adain model may be used however one is already made available in /SSD/ssd/adain
```bash
# for example, train SSD300:
python train.py --config-file configs/your_config_file_here.yaml --AdaIN_model ./ssd/adain/pretrained_adain.pth
```
## TESTING

### Single GPU evaluating

```bash
# for example, evaluate SSD300:
python test.py --config-file configs/your_config_file_here.yaml --ckpt ./outputs/DA_project_baseline/model_final.pth
```

## MISC

Results for each training are saved to the outputs folder by default, if not otherwise specified in the config file.

## USEFUL LINKS

### Models Used

[SSD Baseline](https://drive.google.com/file/d/1TuOw-bpjTKeEvv6SoH13ebhIGjEyt9E0/view?usp=sharing "Google's Homepage")

This should be put inside SSD/ssd/adain/
[Adain Model](https://drive.google.com/file/d/1hF72ulvKh3M7sQgRTM0ulV2aAZteGick/view?usp=sharing)


