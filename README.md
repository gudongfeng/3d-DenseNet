# 3D-DenseNet with TensorFlow (Train in AWS sagemaker)

## Get started !!
### Dependencies
- Tensorflow 1.11
- python 3.6.5
- opencv-python 3.4.3.18
- Pillow 5.3.0 

### Data preparation
1. Download the video dataset and make sure it has the following folder structure (`../video/<action_name>/<video1.avi>` KTH ex: ../kth_video/boxing/person01_boxing_d1_uncomp.avi)
2. Run the `prepare_data_main.py`. You need to specify the `data_dir`, `train_output_path`, and `eval_output_path`. 
  - `data_dir`: `../kth_video`
3. When the script finished. It will print out the following informations
  - Total clips in train dataset: `AAAA` (Take a record of this number)
  - Total clips in eval dataset: `BBBB` (Take a record of this number)

### Train (Local)
1. Paste the number `AAAA` from previous step to `train_total_video_clip` in the `debug_train.py` file.
2. Paste the number `BBBB` from previous step to `eval_total_video_clip` in the `debug_train.py` file.
3. Copy and paste the `eval.tfrecord` and `train.tfrecord` file generated from the previous step to a folder named `../tfrecord`.
4. Set the `DATA_DIR` in the `debug_train.py` to the proper folder name in the previous step.
5. Run `python debug_train.py` (Make sure you have all the dependencies).

### Train (AWS sagemaker)
(To be continued)

## Background
Expand the `Densely Connected Convolutional Networks [DenseNets](https://arxiv.org/abs/1608.06993) to 3D-DenseNet for action recognition (video classification):

- 3D-DenseNet - without bottleneck layers
- 3D-DenseNet-BC - with bottleneck layers

Each model can be tested on such datasets:

- [KTH](http://www.nada.kth.se/cvap/actions/)
- [MERL](http://www.merl.com/demos/merl-shopping-dataset)

A number of layers, blocks, growth rate, video normalization and other training params may be changed trough shell or inside the source code.

There are also many [other implementations](https://github.com/liuzhuang13/DenseNet), they may be useful also.

## Result

Test results on MERL shopping dataset. Video normalization per channels was used.
![image](/fig/result.png)


Approximate training time for models on GeForce GTX TITAN X (12 GB memory):

- 3D-DenseNet(*k* = 12, *d* = 20) - 25 hrs

## Reference

[Thesis](https://ruor.uottawa.ca/bitstream/10393/36739/1/Gu_Dongfeng_2017_thesis.pdf)
