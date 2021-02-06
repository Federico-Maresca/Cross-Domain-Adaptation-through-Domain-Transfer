#!/usr/bin/env bash


#!/bin/sh

#
# a simple way to parse shell script arguments
# 
# please edit and use to your hearts content
# 

ENVIRONMENT="dev"
DT_PATH="/data/db"
SSD_PATH="/data/ssd"
STYLE_PATH="/data/style"
function usage()
{
    echo "this file receives"
    echo ""
    echo "./simple_args_parsing.sh"
    echo "\t-h --help"
    echo "\t --wk-path <working directory here>"
    echo "Dataset folder is implied to be inside the same folder as SSD"
}

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        --wk-path)
            WK_PATH=$VALUE
            ;;    
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done
DT_PATH=$WK_PATH/dataset
SSD_PATH=$WK_PATH/SSD
STYLE_PATH=./ssd/adain/pretrained_adain.pth
#clone SSD github

cd SSD_PATH
#no need to download


#wget dataset voc
#wget dataset clip

#Train with cyclegan

python train.py --config-file configs/dt_VOC_finetune_20ep_1e-6lr_batch64.yaml --ckpt ./outputs/DA_project_baseline/model_final.pth
python train.py --config-file configs/dt_VOC_finetune_20ep_1e-5lr_batch64.yaml --ckpt ./outputs/DA_project_baseline/model_final.pth
python train.py --config-file configs/dt_VOC_finetune_20ep_1e-3lr_batch64.yaml --ckpt ./outputs/DA_project_baseline/model_final.pth
python train.py --config-file configs/dt_VOC_finetune_20ep_1e-6lr.yaml --ckpt ./outputs/DA_project_baseline/model_final.pth
python train.py --config-file configs/dt_VOC_finetune_20ep_1e-5lr.yaml --ckpt ./outputs/DA_project_baseline/model_final.pth
python train.py --config-file configs/dt_VOC_finetune_20ep_1e-3lr.yaml --ckpt ./outputs/DA_project_baseline/model_final.pth

#Train with style

python train.py --config-file configs/dt_VOC_finetune_20ep_1e-6lr_batch64_style.yaml --AdaIN_model $STYLE_PATH --ckpt ./outputs/DA_project_baseline/model_final.pth
python train.py --config-file configs/dt_VOC_finetune_20ep_1e-5lr_batch64_style.yaml --AdaIN_model $STYLE_PATH --ckpt ./outputs/DA_project_baseline/model_final.pth
python train.py --config-file configs/dt_VOC_finetune_20ep_1e-3lr_batch64_style.yaml --AdaIN_model $STYLE_PATH --ckpt ./outputs/DA_project_baseline/model_final.pth
python train.py --config-file configs/dt_VOC_finetune_20ep_1e-6lr_style.yaml --AdaIN_model $STYLE_PATH --ckpt ./outputs/DA_project_baseline/model_final.pth
python train.py --config-file configs/dt_VOC_finetune_20ep_1e-5lr_style.yaml --AdaIN_model $STYLE_PATH --ckpt ./outputs/DA_project_baseline/model_final.pth
python train.py --config-file configs/dt_VOC_finetune_20ep_1e-3lr_style.yaml --AdaIN_model $STYLE_PATH --ckpt ./outputs/DA_project_baseline/model_final.pth


