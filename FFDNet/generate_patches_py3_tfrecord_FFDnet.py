import argparse
import glob
from PIL import Image
import PIL
import random
import tensorflow as tf
import time
from utils_py3_tfrecord import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir_label', dest='src_dir', default='../images/train/groundtruth', help='dir of ground truth data')
parser.add_argument('--src_dir_bayer', dest='src_dir_bayer', default='../images/train/compressed_Q', help='dir of interpolated Bayer data')
parser.add_argument('--Q', dest='quantization_step', default='20', help='quantization step ')
parser.add_argument('--save_dir', dest='save_dir', default='./patches', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=50, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=50, help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=128, help='batch size')
parser.add_argument('--augment', dest='DATA_AUG_TIMES', type=int, default=1, help='data augmentation')
# check output arguments
parser.add_argument('--Train_tfrecord', dest='tfRecord_train', default='subpixel_deblocking.tfrecords', help='training record file')
parser.add_argument('--debug', dest='isDebug', type=bool, default=False, help='Debug Mode')
args = parser.parse_args()

def generate_patches():
    isDebug = args.isDebug
    filepaths = sorted(glob.glob(args.src_dir + '/*'))
    filepaths_bayer = sorted(glob.glob(args.src_dir_bayer + args.quantization_step + '/*'))
    if isDebug:
        numDebug = 5
        filepaths = filepaths[:numDebug] # take only ten images to quickly debug
        filepaths_bayer = filepaths_bayer[:numDebug]
    print("number of training images %d" % len(filepaths))
    count = 0 # calculate the number of patches
    for i in range(len(filepaths)):
        img = Image.open(filepaths[i])
        im_h, im_w = img.size
        for x in range(0 + args.step, (im_h - args.pat_size), args.stride):
            for y in range(0 + args.step, (im_w - args.pat_size), args.stride):
                count += 1
    origin_patch_num = count * args.DATA_AUG_TIMES
    if origin_patch_num % args.bat_size != 0:
        numPatches = int(origin_patch_num / args.bat_size) * args.bat_size
    else:
        numPatches = int(origin_patch_num)
    print("Total patches = %d , batch size = %d, total batches = %d" %(numPatches, args.bat_size, numPatches / args.bat_size))
    time.sleep(2)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)    
    count = 0

    # generate patches
    writer = tf.python_io.TFRecordWriter(args.save_dir + '/' + args.tfRecord_train)
    for i in range(len(filepaths)):
        img = Image.open(filepaths[i])
        img_Bayer = Image.open(filepaths_bayer[i])
        img_s = np.array(img, dtype="uint8")
        img_s_Bayer = np.array(img_Bayer, dtype="uint8")
        im_h, im_w, _ = img_s.shape
        print("The %dth image of %d training images" %(i+1, len(filepaths)))
        for j in range(args.DATA_AUG_TIMES):
            for x in range(0 + args.step, im_h - args.pat_size, args.stride):
                for y in range(0 + args.step, im_w - args.pat_size, args.stride):
                    if j == 0:
                        random_seed = 0
                    else:
                        random_seed = random.randint(1, 7)
                    image_label = data_augmentation(img_s[x:x + args.pat_size, y:y + args.pat_size, 0:3], random_seed) # some images have an extra blank channel 
                    image_bayer = data_augmentation(img_s_Bayer[x:x + args.pat_size, y:y + args.pat_size, 0:3], random_seed)
                    image_label = image_label.tobytes()
                    image_bayer = image_bayer.tobytes()
                    count += 1
                    example = tf.train.Example(features = tf.train.Features(feature={
                        "img_label":tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_label])),
                        'img_bayer':tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_bayer]))
                    }))
                    if count<= numPatches:
                        writer.write(example.SerializeToString())
                    else:
                        break
    writer.close()
    print("Total patches = %d , batch size = %d, total batches = %d" %(numPatches, args.bat_size, numPatches / args.bat_size))
    print("Training data has been written into TFrecord.")

if __name__ == '__main__': 
    generate_patches()
