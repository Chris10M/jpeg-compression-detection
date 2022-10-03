# JPEG Image Compression And Quality Factor Detection

Few methods for JPEG Image Compression And Quality Factor Detection


Low Compression            |  Medium Compression       |  High Compression 
:-------------------------:|:-------------------------:|:-------------------------:
![](images/high.png)       |  ![](images/mid.png)      |  ![](images/low.png)


## Methods

### DCT Based JPEG Compression Detection

<p align="center">
<img src="images/dct.gif" alt="method" width="600"/></br>
</p>

* Split the image into 8x8 blocks.
* Compute the DCT coefficients for each block and quantize the frequency components.
* Compute the normalized frequencies across all the blocks.
* Compute the variance of the high frequency blocks.
* If the variance is high, then unclamped high frequency components are present. If the variance is low, then jpeg quanitzed high frequency components are present.
* The above observation can be used to classify if a jpeg image is compressed. 

### Model Based Detection and Regression


* Compute a reference denoised image using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
* Compute the residual image, i.e., difference between reference denoised image and input image.
* The residual image contains high-frequency components, artifacts, etc.
* The residual image is passed to a resnet model to get classifcation and regression of jpeg-compression and quality factor respectively.
* The model is trained with augmenting single, doube jpeg compressions.
* Non-aligned double JPEG compression should be implicitly handled as the reference image should produce a constant noise pattern. 


<p align="center">
<img src="images/method_overview.png" alt="method" width="600"/></br>
</p>

### LRW 
Alignment Plot                      |  Melspectogram Output          
:-------------------------:|:-------------------------:|
![](images/attention.png)       |  ![](images/meloutput.png)  


## Usage

### Demo

The pretrained model is available [here](https://www.mediafire.com/file/evktjxytts2t72c/lip2speech_final.pth/file) [265.12 MB]

Download the pretrained model and place it inside **savedmodels** directory. To visulaize the results,  we run demo.py.

```
python3 demo.py
``` 

#### Default arguments

* dataset: LRW (10 Samples)
* root: Datasets/SAMPLE_LRW
* model_path: savedmodels/lip2speech_final.pth
* encoding: voice


### Evaluate 

Evaluates the ESTOI score for the given Lip2Speech model. (Higer is better)

```
python3 evaluate.py --dataset LRW --root Datasets/LRW --model_path savedmodels/lip2speech_final.pth
```


### Train

To train the model, we run train.py

```
python3 train.py --dataset LRW --root Datasets/LRW --finetune_model_path savedmodels/lip2speech_final.pth
```

* finetune_model_path - Use as base model to finetune to dataset. (optional)



## Acknowledgement

[tacotron2](https://github.com/NVIDIA/tacotron2)


## Citation

If you use this software in your work, please cite it using the following metadata.


```
@software{Millerdurai_Lip2Speech_2021,
author = {Millerdurai, Christen and Abdel Khaliq, Lotfy and Ulrich, Timon},
month = {8},
title = {{Lip2Speech}},
url = {https://github.com/Chris10M/Lip2Speech},
version = {1.0.0},
year = {2021}
}
