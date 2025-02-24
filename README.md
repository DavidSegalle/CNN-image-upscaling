# U-Net Image Upscaling

The goal of this project is to upscale images from 320x240 to 640x480 while also bringing image quality up to standart to the new resolution.

In order to achieve this I have used a U-net architecture with convolutional and Leaky ReLu layers. The dataset is a subset of coco containing only the images in the desired output resolution. More information on the network setup can be found in _model_plot.png_

## Setting up the dataset

Download coco dataset, make a copy to the correct place, remove images outside desired resolution and make low_res copies for training.

``` bash
cd src
python3 download_dataset.py
```

## Training the model

``` bash
cd src
python3 train_model.py
```

## Running the model

``` bash
cd src
python3 run_model.py
```
