import gc
import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from msssim import MultiScaleSSIM

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

def read_and_decode(filename, batch_size, patch_size):
    filename_queue = tf.train.string_input_producer([filename])#create a queue
    reader = tf.TFRecordReader()#read *.tfrecords
    _, serialized_example = reader.read(filename_queue)#return filename and file
    features = tf.parse_single_example(serialized_example, features = {
                                        'img_label':tf.FixedLenFeature([], tf.string),
                                        'img_bayer':tf.FixedLenFeature([], tf.string), 
                                        })#take out image and label
    img_bayer = tf.decode_raw(features['img_bayer'], tf.uint8)
    img_bayer = tf.reshape(img_bayer, [patch_size, patch_size, 3])
    img_label = tf.decode_raw(features['img_label'], tf.uint8)
    img_label = tf.reshape(img_label, [patch_size, patch_size, 3])
    img_label = tf.cast(img_label, tf.float32)
    img_bayer = tf.cast(img_bayer, tf.float32)
    img_labelBatch ,img_bayerBatch = tf.train.shuffle_batch([img_label, img_bayer], batch_size=batch_size, capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
    return img_labelBatch, img_bayerBatch

def load_images(filelist):
    if not isinstance(filelist, list):
        im = Image.open(filelist)
        return np.array(im).reshape(1, im.size[1], im.size[0], 3)
    data = []
    for file in filelist:
        im = Image.open(file)
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 3))
    return data

def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8'))
    im.save(filepath)

def cal_psnr(im1, im2):
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr

def tf_psnr(im1, im2):
    mse = tf.losses.mean_squared_error(labels=im2, predictions=im1)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))

def imcpsnr(im1, im2, peak=255, b=0):
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    im1 = im1[b:im1.shape[0]-b,b:im1.shape[1]-b,:]
    im2 = im2[b:im2.shape[0]-b,b:im2.shape[1]-b,:]
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    return 10 * np.log10(255 ** 2 / mse)

def impsnr(im1, im2, peak=255, b=0):
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    im1 = im1[b:im1.shape[0]-b,b:im1.shape[1]-b,:]
    im2 = im2[b:im2.shape[0]-b,b:im2.shape[1]-b,:]
    csnr = np.ones(im1.shape[2])
    for i in range(im1.shape[2]):
        mse = ((im1[:,:,i].astype(np.float) - im2[:,:,i].astype(np.float)) ** 2).mean()
        csnr[i] = 10 * np.log10(255 ** 2 / mse)
    return csnr

def getBayer(RGB_Image):
    RGB_Image = np.squeeze(RGB_Image)
    h, w, _ = RGB_Image.shape
    Bayer = np.zeros([h, w], dtype=np.uint8)
    Bayer[0:h:2,0:w:2] = RGB_Image[0:h:2,0:w:2,0]#R
    Bayer[1:h:2,0:w:2] = RGB_Image[1:h:2,0:w:2,1]#G1
    Bayer[0:h:2,1:w:2] = RGB_Image[0:h:2,1:w:2,1]#G2
    Bayer[1:h:2,1:w:2] = RGB_Image[1:h:2,1:w:2,2]#B
    return Bayer

def MS_SSIM(im1, im2, b=0):
    im1 = im1[:,b:im1.shape[1]-b,b:im1.shape[2]-b,:]
    im2 = im2[:,b:im2.shape[1]-b,b:im2.shape[2]-b,:]
    return MultiScaleSSIM(im1, im2)

def SSIM(im1, im2, b=0):
    im1 = tf.convert_to_tensor(im1[:,b:im1.shape[1]-b,b:im1.shape[2]-b,:], dtype=np.float32)
    im2 = tf.convert_to_tensor(im2[:,b:im2.shape[1]-b,b:im2.shape[2]-b,:], dtype=np.float32)
    return tf.image.ssim(im1, im2, max_val=255)
