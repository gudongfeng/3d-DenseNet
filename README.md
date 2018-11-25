# 3D-DenseNet with TensorFlow (Train in AWS sagemaker)

## Get started !!
### Dependencies
- Tensorflow 1.11
- python 3.6.5
- opencv-python 3.4.3.18
- Pillow 5.3.0 
- sagemaker 1.15.2


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
**Note: It turns out Sagemaker doesn't support python3 for Tensorflow script at this moment (2018.Nov.1)!!!**
**So I will stop working on this part and left the `sagemaker_main.template` file as it is for now.**
1. Register AWS account. [AWS Console](https://console.aws.amazon.com)
2. Create an IAM user with only `Programmatic access` and attached `AmazonS3FullAccess` and `AmazonSageMakerFullAccess` to this IAM user. Keep a record of your `Access Key ID` and `Secret Access Key` (**Don't tell anyone this information!!! Even your husband/wife**).
3. Install [boto3](https://aws.amazon.com/sdk-for-python/) on your local desktop. Run `aws configure` in your console and paste the `Access Key ID` and `Secret Access Key` from previous step. Keep in mind the region (ex: `us-west-2`) that you used.
4. Create a new Role with name `sagemaker-full-access-role` and attach an inline policy with the following [JSON](http://gudongfeng.me/sagemaker-role-inline-policy.txt)
5. Create a new S3 bucekt with whatever name you want in the same region in Step3. Let said the S3 bucket name is `machine_leaning_data_bucket`.
6. Rename the `sagemaker_main.template` to `sagemaker_main.py`
7. Copy the new Role ARN (ex: `arn:aws:iam::<aws_account_id>:role/sagemaker-full-access-role`) and paste it to the `role` value in the `sagemaker_main.py`
8. Replace the `<s3_bucket_name>` in `sagemaker_main.py` with S3 bucket name `machine_leaning_data_bucket` (Whatever S3 bucket name you have).
9. Chooes one option in the `sagemaker_main.py` and run `python sagemaker_main.py`. Notice that if you choose 
> As I said at the beginning, sagemaker doesn't support tensorflow docker image with python version 3, so you will get error `Attempted relative import in non-package` at this moment. I will try to rework this file once sagemaker support it. 

## Background
Expand the `Densely Connected Convolutional Networks [DenseNets](https://arxiv.org/abs/1608.06993) to 3D-DenseNet for action recognition (video classification):

- 3D-DenseNet - without bottleneck layers
- 3D-DenseNet-BC - with bottleneck layers

Each model can be tested on such datasets:

- [KTH](http://www.nada.kth.se/cvap/actions/)
- [MERL](http://www.merl.com/demos/merl-shopping-dataset)

A number of layers, blocks, growth rate, video normalization and other training params may be changed trough shell or inside the source code.

There are also many [other implementations](https://github.com/liuzhuang13/DenseNet), they may be useful also.

## Reference

[Thesis](https://ruor.uottawa.ca/bitstream/10393/36739/1/Gu_Dongfeng_2017_thesis.pdf)
