# PELEscores

## The official code for PELEcores. 

## Requirement:
```python
python >= 3.6
torch = 1.10.1
```
## Data：
Please proceed to obtain your own qualification and download:
### CT：
https://github.com/MIRACLE-Center/CTPelvic1K

### Xray:
cgmh-pelvisseg: https://www.kaggle.com/datasets/tommyngx/cgmh-pelvisseg

PXR150: https://www.nature.com/articles/s41467-021-21311-3

Pleas feel free to contact me if you have any problem: 123466188[at]qq.com

## How to run
You should follow the steps described in detail in our paper. Specifically, we have also set up four steps in this warehouse.

### step0 & pre-step：deepdrr
In this way, you should generate DRR images by deepdrr using our CT data.
you should Install deepdrr according to the instructions: https://github.com/arcadelab/deepdrr.
```python
cd 0deepdrr
python example.py
# you can also create your own projection file, take it easy.
```
### step1：cyclegan between xray and drr images
You also need to configure your cyclegan environment according to the instructions：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
```python
cd 1cycleganxray2drr
train phase:
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --preprocess scale_width_and_crop --load_size 1024 --crop_size 360
test phase: 
python test.py --dataroot ./datasets/maps/testA --name maps_cyclegan --model test --no_dropout --preprocess scale_width --load_size 1024
```
### step2: unet & nnunet
In this phase, you can use different models set by ourselves to get best performance.
For nnunet setting and instructions, please refer to: https://github.com/MIC-DKFZ/nnUNet/tree/master
```python
cd 2Pytorch-UNet-master_trainbonemask
train phase:
python train.py
test phase:
python test.py
```
### step3：landmark detection
```python
pip3 install -r requirements.txt
train phase:
python3 main.py -d ../runs -r unet2d_runs -p train -m unet2d -e 100
test phase:
python3 main.py -d ../runs -r GU2Net_runs -p test -m gln -l u2net -c CHECKPOINT_PATH
evaluation:
python3 evaluation.py -i ../runs/GU2Net_runs/results/test_epochxxx
```



