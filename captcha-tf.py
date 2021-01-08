# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
import os
import numpy as np
import pandas as pd
import json
import glob
from PIL import Image

N_LABELS = 10
H, W, C = 100, 120, 3
D = 4

def model(dir):
    
    input_layer = tf.keras.Input(shape=(H, W, C))
    x = layers.Conv2D(16, 3, activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)

    x = layers.Dense(D * N_LABELS, activation='softmax')(x)
    x = layers.Reshape((D, N_LABELS))(x)

    model = models.Model(inputs=input_layer, outputs=x)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    
    df,train_idx,valid_idx = _load_data(dir)
    batch_size = 64
    valid_batch_size = 64
    train_gen = get_data_generator(df, train_idx, for_training=True, batch_size=batch_size)
    valid_gen = get_data_generator(df, valid_idx, for_training=True, batch_size=valid_batch_size)

    history = model.fit(train_gen,
                        steps_per_epoch=len(train_idx)//batch_size,
                        epochs=5,
                        validation_data=valid_gen,
                        validation_steps=len(valid_idx)//valid_batch_size)
    return model

def parse_filepath(filepath):
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        label, _ = filename.split("_")
        return label
    except Exception as e:
        print('error to parse %s. %s' % (filepath, e))
        return None, None

def _load_data(dir):
    files = glob.glob(os.path.join(dir, "*.png"))
    attributes = list(map(parse_filepath, files))
    df = pd.DataFrame(attributes)
    df['file'] = files
    df.columns = ['label', 'file']
    df = df.dropna()
    p = np.random.permutation(len(df))
    train_up_to = int(len(df) * 0.7)
    train_idx = p[:train_up_to]
    test_idx = p[train_up_to:]

    # split train_idx further into training and validation set
    train_up_to = int(train_up_to * 0.7)
    train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
    print('train count: %s, valid count: %s, test count: %s' % (len(train_idx), len(valid_idx), len(test_idx)))
    return df,train_idx,valid_idx


def get_data_generator(df, indices, for_training, batch_size=16):
    images, labels = [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, label = r['file'], r['label']
            im = Image.open(file)
            im = np.array(im) / 255.0
            images.append(np.array(im))
            labels.append(np.array([np.array(to_categorical(int(i), N_LABELS)) for i in label]))
            if len(images) >= batch_size:
                yield np.array(images), np.array(labels)
                images, labels = [], []
        if not for_training:
            break
    
def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    captcha_classifier = model(args.train)
    if args.current_host == args.hosts[0]:
        captcha_classifier.save(os.path.join(args.sm_model_dir , '000000001'), 'my_model.h5')