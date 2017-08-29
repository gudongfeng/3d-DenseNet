# 3D-DenseNet with TensorFlow

Expand the `Densely Connected Convolutional Networks [DenseNets](https://arxiv.org/abs/1608.06993) to 3D-DenseNet for action recognition (video classification):

- 3D-DenseNet - without bottleneck layers
- 3D-DenseNet-BC - with bottleneck layers

Each model can be tested on such datasets:

- [KTH](http://www.nada.kth.se/cvap/actions/)
- [MERL](http://www.merl.com/demos/merl-shopping-dataset)

A number of layers, blocks, growth rate, video normalization and other training params may be changed trough shell or inside the source code.

There are also many [other implementations](https://github.com/liuzhuang13/DenseNet), they may be useful also.

## Pre-request libraries
- python2
- tensorflow 1.0
- opencv2 for python2

## Step 1: Data preparation (UCF dataset example)

1. Download the [UCF101](http://crcv.ucf.edu/data/UCF101/UCF101.rar) (Action Recognition Data Set).
2. Extract the `UCF101.rar` file and you will get `../UCF101/<action_name>/<video_name.avi>` folder structure.
3. Use the `./data_prepare/convert_video_to_images.sh` script to decode the `UCF101` video files to image files.
    - run `./data_prepare/convert_video_to_images.sh ../UCF101 25` (number `25` means the fps rate)
4. Use the `./data_prepare/convert_images_to_list.sh` script to create/update the `{train,test}.list` according to the new `UCF101` image folder structure generated from last step (from images to files).
    - run `./data_prepare/convert_images_to_list.sh .../UCF101 4`, this will update the `test.list` and `train.list` files (number `4` means the ratio of test and train data is 1/4)
    - `train.list` example:
        ```
        ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01 0
        ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c02 0
        ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03 0
        ApplyLipstick/v_ApplyLipstick_g01_c01 1
        ApplyLipstick/v_ApplyLipstick_g01_c02 1
        ApplyLipstick/v_ApplyLipstick_g01_c03 1
        Archery/v_Archery_g01_c01 2
        Archery/v_Archery_g01_c02 2
        Archery/v_Archery_g01_c03 2
        Archery/v_Archery_g01_c04 2
        BabyCrawling/v_BabyCrawling_g01_c01 3
        BabyCrawling/v_BabyCrawling_g01_c02 3
        BabyCrawling/v_BabyCrawling_g01_c03 3
        BabyCrawling/v_BabyCrawling_g01_c04 3
        BalanceBeam/v_BalanceBeam_g01_c01 4
        BalanceBeam/v_BalanceBeam_g01_c02 4
        BalanceBeam/v_BalanceBeam_g01_c03 4
        BalanceBeam/v_BalanceBeam_g01_c04 4
        ...
        ```
5. Copy/Cut the `test.list` and `train.list` files to the root of video folder (`../UCF101`).

## Step 2: Train or Test the model

- Check the trainig help message

    `python run_dense_net_3d.py -h`

- Train and test the program

    `python run_dense_net_3d.py --train --test -ds path/to/video_folder` \
    `// Notices that all the logs message will be written in log.txt file in the root folder`


## Options

- `run_dense_net_3d.py` -> `train_params` settings
    ```
    'num_classes': 5,               # The number of the classes that this dataset had
    'batch_size': 10,               # Batch Size When we trian the model
    'n_epochs': 100,                # The total number of epoch we run the model
    'crop_size': (64,64),           # The (width, height) of images that we used to trian the model
    'sequence_length': 16,          # The length of the video clip
    'overlap_length': 8,            # The overlap of the images when we extract the video clips,
                                      this should be less than sequence_length
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 50,        # epochs * 0.5
    'reduce_lr_epoch_2': 75,        # epochs * 0.75
    'validation_set': True,         # Whether used validation set or not
    'validation_split': None,       # None or float
    'queue_size': 300,              # The data queue size when we extract the data from dataset,
                                      should be set according to your memory size
    'normalization': 'std',         # None, divide_256, divide_255, std
    ```


## Result

Test results on MERL shopping dataset. Video normalization per channels was used.
![image](/fig/result.png)


Approximate training time for models on GeForce GTX TITAN X (12 GB memory):

- 3D-DenseNet(*k* = 12, *d* = 20) - 25 hrs

