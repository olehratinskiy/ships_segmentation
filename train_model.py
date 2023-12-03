import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, concatenate
import keras.backend as K
from sklearn.model_selection import train_test_split


def convert_mask_data(data):
    """
    Creates an empty image, goes through encoded pixels of mask and
    adds them as white on empty black image
    :param data: list with mask encoded pixels
    :return: mask image
    """
    # creation of empty black image
    empty_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)
    local_mask_data = [int(i) for i in data.split()]

    # for each pair (pixel and length of sequence) add pixels on empty_mask
    for i in range(0, len(local_mask_data) - 1, 2):
        line = round(local_mask_data[i] / IMG_WIDTH)
        col_start = local_mask_data[i] - (line * IMG_WIDTH)
        col_end = col_start + local_mask_data[i + 1]
        empty_mask[col_start: col_start + local_mask_data[i + 1] + 1, line - 1, :] = 255

    return empty_mask


def dice_score(original, predicted):
    """
    Calculates dice score between original mask and predicted one to see if they are similar
    :param original: original mask
    :param predicted: predicted mask
    :return: score: float value in range [0, 1] where 0 means they are totaly different and 1 - same
    """
    # calcultes result according to dice score formula
    doubled_intersection = 2 * np.sum(original * predicted)
    masks_sum = np.sum(original) + np.sum(predicted)
    score = doubled_intersection / masks_sum
    return score


def convolutional_block(filters_num, input_layer):
    """
    Creates convolutional block which is a part of UNet.
    Convolutional layers are needed to extract and learn features from input image.
    :param filters_num: int value, number of filters in convolutional layers
    :param input_layer: data before applying convolutional layers
    :return: data after applying convolutional layers
    """
    conv = Conv2D(filters=filters_num, kernel_size=3, padding='same', activation='relu')(input_layer)
    conv = Conv2D(filters=filters_num, kernel_size=3, padding='same', activation='relu')(conv)
    return conv


def encoder(filters_num, input_layer):
    """
    Creates encoder blocks, which are used in the left part of UNet. It uses conv_block and
    MaxPool2D to make image smaller.
    :param filters_num: int value, number of filters in convolutional layers
    :param input_layer: tensor before applying encoder
    :return: tensor after applying encoder
    """
    conv = convolutional_block(filters_num, input_layer)
    enc = MaxPool2D(pool_size=(2, 2))(conv)
    return conv, enc


def decoder(filters_num, current_input, prev_input):
    """
    Creates decoder blocks, which are used in the right part of UNet.
    It uses UpSampling2D to increase size of image. Then, according to UNet architecture,
    concatenates conv_block on the same level on the left part of UNet and current tensor. After that applies
    conv_block.
    :param filters_num: int value, number of filters in convolutional layers
    :param current_input: tensor before applying decoder
    :param prev_input: tensor after applying conv_block on the same level on the left part of UNet
    :return: tensor after applying decoder
    """
    dec = UpSampling2D((2, 2))(current_input)
    conv = concatenate([dec, prev_input])
    conv = convolutional_block(filters_num, conv)
    return conv


# load masks encoded data and set constant size of images (original, masks, predicted)
K.clear_session()
IMG_HEIGHT, IMG_WIDTH = 768, 768
train_masks_data = pd.read_csv('data/train_ship_segmentations_v2.csv')

# preprocess dataframe and concatenate all masks data if they refer to the same original image (according to ImageId)
train_masks_data['EncodedPixels'] = train_masks_data['EncodedPixels'].astype(str)
train_masks_data['EncodedPixels'] = train_masks_data['EncodedPixels'].fillna('')
train_masks_data['EncodedPixels'] = train_masks_data['EncodedPixels'].replace('nan', '')
train_masks_data = train_masks_data.groupby('ImageId')['EncodedPixels'].agg(lambda i: ' '.join(i)).reset_index()

# load original images
folder = 'data/train_v2/'
img_list = []

for filename in sorted(os.listdir(folder))[:300]:
    img_path = os.path.join(folder, filename)
    img_list.append(cv2.imread(img_path))

X = np.array(img_list)

# convert encoded masks data to masks images
masks_list = []

for mask_data in train_masks_data['EncodedPixels'][:300]:
    masks_list.append(convert_mask_data(mask_data))

y = np.array(masks_list)


# build UNet model
# input layer
inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))

# encoders
conv_1, enc_1 = encoder(64, inputs)
conv_2, enc_2 = encoder(128, enc_1)

# middle part of UNet
middle = convolutional_block(256, enc_2)

# decoders
dec_1 = decoder(128, middle, conv_2)
dec_2 = decoder(64, dec_1, conv_1)

# output layer with 3 channels and activation softmax
outputs = Conv2D(3, 1, padding='same', activation='softmax')(dec_2)

unet_model = Model(inputs, outputs)
optimizer = keras.optimizers.Adam(lr=0.01)

# model compilation with definaed optimiter and binary_crossentropy loss function
unet_model.compile(optimizer=optimizer, loss='binary_crossentropy')
unet_model.summary()

# normalization of data
X = X.astype("float32") / 255
y = y.astype("float32") / 255

# splitting data into train and test
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# model training
history = unet_model.fit(train_X, train_y, batch_size=2, epochs=2)

# save model and test data to files
unet_model.save('unet_model.h5')

np.save('data/test_X.npy', test_X)
np.save('data/test_y.npy', test_y)

