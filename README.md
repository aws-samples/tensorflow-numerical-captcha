# Solving Captcha

This repository contains example notebook that shows how to apply deep learning in Amazon SageMaker.

## Introduction

This example demonstrate how to use Amazon SageMaker to build, train and deploy TensorFlow based model. In this example a deep learning model is trained to infer 4 digit numerical captcha.

## Setup Information

There are three code files in this repoistory.

1. `Numeric-Captcha-SageMaker-TensorFlow.ipynb` - Notebook containing code for generating test data, train, deploy and test the model.
2. `gen_captcha.py` - python script to genrate images for training and test purposes. Refer next section usage to know how to use it to generated different dataset. This script uses "captcha" library that generates audio and image CAPTCHAs. In this case only image captchas are considered.
3. `captcha-tf.py` - Tensorflow based model training script which is be used to train model in SageMaker. The following training step is using GPU based instance, i.e. ml.p3.2xlarge. It takes around 10 minutes training time  with training data generated in previous steps (6 permutations). For traing dataset generated out of 60 permutations takes around one hour to train the model. Refer available instance types [Available Instance Types](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html) to choose different instance type. Refer [Amazon SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/) to know the pricing.

### Note

- To run notbook instance in Amazon SageMaker Studio you can choose any of default "fast launch" instance else if want to run it on SageMaker Notebook instances, you can use latest series of standard instances, for example - ml.m5.large or ml.m5.xlarge.

- Set the kernel as "Python 3 (TensorFlow 2 CPU Optimized)"

- If you are going to use SageMaker Notbook Instance, please follow steps [Create a Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html)

- If you are going to use SageMaker Studio first time to run this code, you need to complete the Amazon SageMaker Studio onboarding process.For more information, see [Onboard to Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html)

### Dependencies

captcha 0.3

TensorFlow 2.1.0

## Data Generation

Data generattion script is included as `gen_captcha.py`. Script can be used to generate datasets based on different permutations. This step is included as first steps in notbook. To generate larger dataset, generate dataaet with (for example) 60 permutations.

### Usage

To Generate 4 digits captcha with 6 permutations

`$ python gen_captcha.py -d --npi=4 -n 6`

To Generate 4 digits captcha with 60 permutations

`$ python gen_captcha.py -d --npi=4 -n 60`

Transfer the generated datasets to Amazon S3 bucket to use it during model training.

## Next Steps  

Follow the notebook `Numeric-Captcha-SageMaker-TensorFlow` to build, train and deploy model.
