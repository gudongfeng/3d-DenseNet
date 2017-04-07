# 3D-DenseNet with TensorFlow

Two types of `Densely Connected Convolutional Networks [DenseNets](https://arxiv.org/abs/1608.06993) are available:

- DenseNet - without bottleneck layers
- DenseNet-BC - with bottleneck layers

Each model can be tested on such datasets:

- [UCF101](http://crcv.ucf.edu/data/UCF101/UCF101.rar)
- [MERL](http://www.merl.com/demos/merl-shopping-dataset)

A number of layers, blocks, growth rate, video normalization and other training params may be changed trough shell or inside the source code.

There are also many [other implementations](https://github.com/liuzhuang13/DenseNet) they may be useful also.

Citation for DenseNet:

     @article{Huang2016Densely,
            author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
            title = {Densely Connected Convolutional Networks},
            journal = {arXiv preprint arXiv:1608.06993},
            year = {2016}
     }

## Step 1: Data preparation

1. Download the [UCF101](http://crcv.ucf.edu/data/UCF101/UCF101.rar) (Action Recognition Data Set).
2. Extract the `UCF101.rar` file and you will get `UCF101/<action_name>/<video_name.avi>` folder structure.
3. Use the `./data_prepare/convert_video_to_images.sh` script to decode the `UCF101` video files to image files.
    - run `./data_prepare/convert_video_to_images.sh ../UCF101 5` (number `5` means the fps rate)
4. Use the `./data_prepare/convert_images_to_list.sh` script to create/update the `{train,test}.list` according to the new `UCF101` image folder structure generated from last step (from images to files).
    - run `./data_prepare/convert_images_to_list.sh .../UCF101 4`, this will update the `test.list` and `train.list` files (number `4` means the ratio of test and train data is 1/4)
    - `train.list`:
        ```
        database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01 0
        database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c02 0
        database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03 0
        database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c01 1
        database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c02 1
        database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c03 1
        database/ucf101/train/Archery/v_Archery_g01_c01 2
        database/ucf101/train/Archery/v_Archery_g01_c02 2
        database/ucf101/train/Archery/v_Archery_g01_c03 2
        database/ucf101/train/Archery/v_Archery_g01_c04 2
        database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c01 3
        database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c02 3
        database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c03 3
        database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c04 3
        database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c01 4
        database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c02 4
        database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c03 4
        database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c04 4
        ...
        ```
5. Copy/Cut the `test.list` and `train.list` files to the `data_providers` folders.

## Step 2: Train or Test the model

- Train the program

    python run_dense_net_3d.py --train --test --dataset=MERL

- Check parameter help message

    python run_dense_net_3d.py --help

## Options

- `run_dense_net_3d.py` -> `train_params_<dataset>` settings
    ```
    'num_classes': 5,               # The number of the classes that this dataset had
    'batch_size': 10,               # Batch Size When we trian the model
    'n_epochs': 100,                # The total number of epoch we run the model
    'crop_size': 64,                # The weight and length of images that we used to trian the model
    'sequence_length': 16,          # The length of the video clip
    'overlap_length': 8,            # The overlap of the images when we extract the video clips,
                                      this should be less than sequence_length
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 50,        # epochs * 0.5
    'reduce_lr_epoch_2': 75,        # epochs * 0.75
    'validation_set': True,         # Whether used validation set or not
    'validation_split': None,       # None or float
    'shuffle': True,                # None, once_prior_train, every_epoch
    'normalization': 'by_channels', # None, divide_256, divide_255, by_channels
    ```


## Result

Test results on MERL dataset. Video normalization per channels was used.

|Model type             |Depth  |MERL      |
|-----------------------|:-----:|---------:|
|DenseNet(*k* = 12)     |20     |70%       |


Approximate training time for models on GeForce GTX TITAN X (12 GB memory):

- DenseNet(*k* = 12, *d* = 20) - 25 hrs
- DenseNet-BC(*k* = 12, *d* = 100) - 1 day 18 hrs


## Dependencies

- Model was tested with Python 3.4.3+ and Python 3.5.2 with and without CUDA.
- Model should work as expected with TensorFlow >= 0.10. Tensorflow 1.0 support was included.

Repo supported with requirements file - so the easiest way to install all just run ``pip install -r requirements.txt``.

