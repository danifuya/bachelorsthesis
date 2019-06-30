import argparse
from glob import glob

import tensorflow as tf
import math

from model_py3_tfrecord_FastARCNN import denoiser
from utils_py3_tfrecord import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--Q', dest='quantization_step', default='20', help='quantization step ')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=math.pow(10,-3), help='initial learning rate for adam')
#parser.add_argument('--lr2', dest='lr2', type=float, default=5*math.pow(10,-5), help='initial learning rate for adam')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=24, help='patch size')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu', dest='num_gpu', type=str, default="0", help='choose which gpu')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--eval_set', dest='eval_set', default='test', help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='deblocking', help='dataset for testing')
args = parser.parse_args()

#weigth decay momentum optimizer
#L2 regularization
#tensorboard

def denoiser_train(denoiser, lr, eval_every_step, patch_size):
    img_labelBatch, img_bayerBatch = read_and_decode('./patches/subpixel_deblocking.tfrecords', args.batch_size, patch_size)
    eval_files_gt = glob('../images/{}/groundtruth/*'.format(args.eval_set))
    eval_data_gt = load_images(eval_files_gt)
    eval_files_bl = glob(('../images/{}/compressed_Q' + args.quantization_step +'/*').format(args.eval_set))
    eval_data_bl = load_images(eval_files_bl)
    denoiser.train(img_labelBatch, img_bayerBatch, eval_data_gt, eval_data_bl, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, lr=lr, sample_dir=args.sample_dir, eval_every_step=eval_every_step)

def denoiser_test(denoiser):
    test_files_gt = glob('../images/{}/groundtruth/*'.format(args.test_set))
    test_files_bl =  glob(('../images/{}/compressed_Q' + args.quantization_step +'/*').format(args.test_set))
    denoiser.test(test_files_gt, test_files_bl, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)

def ensemble_test(denoiser):
    test_files_gt = glob('../test/{}/groundtruth/*'.format(args.test_set))
    test_files_bl = glob('../test/{}/compressed/*'.format(args.test_set))
    denoiser.self_ensemble_test(test_files_gt, test_files_bl, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)

def main(_):
    # check the path of checkpoint, samples and test
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    if args.use_gpu:
        print("GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu
        gpu_options = tf.GPUOptions(allow_growth = True) #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess, batch_size = args.batch_size, patch_size=args.patch_size)
            if args.phase == 'train':
                numPatches = 0
                for record in tf.python_io.tf_record_iterator('./patches/subpixel_deblocking.tfrecords'):
                    numPatches += 1
                # learning rate strategy
                iter_epoch = numPatches//args.batch_size
                iter_all = args.epoch*iter_epoch
                lr = args.lr * np.ones(iter_all) 
                lr[iter_epoch*10:] = lr[0] / 2.0
                lr[iter_epoch*20:] = lr[0] / 10.0
                lr[iter_epoch*30:] = lr[0] / 20.0
                lr[iter_epoch*40:] = lr[0] / 100.0
                               
               # for epoch in range (1, args.epoch):
                 #   lr[iter_epoch*epoch:] = lr[0] * math.pow(10.0,-(3.0/49.0)*epoch)
                denoiser_train(model, lr=lr, eval_every_step=iter_epoch, patch_size=args.patch_size)
            elif args.phase == 'test':
                denoiser_test(model)
            elif args.phase == 'ensemble':
                ensemble_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        with tf.Session() as sess:
            model = denoiser(sess, batch_size = args.batch_size, patch_size=args.patch_size)
            if args.phase == 'train':
                denoiser_train(model, lr=lr, eval_every_step=iter_epoch, patch_size=args.patch_size)
            elif args.phase == 'test':
                denoiser_test(model)
            elif args.phase == 'ensemble':
                ensemble_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)

if __name__ == '__main__':
    tf.app.run()