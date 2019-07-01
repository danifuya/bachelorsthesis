import time

from utils_py3_tfrecord import *
from model_database_FastARCNN import *

class denoiser(object):
    def __init__(self, sess, input_c_dim=3, batch_size=64, patch_size=160):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='GroundTruth') # ground truth
        self.X = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='BilinearInitialization') # input of the network
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
       #self.lr2 = tf.placeholder(tf.float32, name='learning_rate2')
        
        #self.Y = subpixel_new(self.X, is_training=self.is_training)
        self.Y = subpixel_new(self.X, is_training=self.is_training)
       #loss has to be mean squared error
        self.lossRGB = (1.0 /batch_size / patch_size / patch_size) * tf.nn.l2_loss(self.Y_ -self.Y)
        self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([self.lossRGB] + self.reg_losses)

        self.eva_psnr = tf_psnr(self.Y, self.Y_)

        #optimizer=tf.train.MomentumOptimizer(self.lr, momentum=0.9)
        #optimizer = tf.train.GradientDescentOptimizer(self.lr, name='GradientDescent')
        optimizer=tf.train.AdamOptimizer(self.lr, name='Adam')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for BN?
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, test_data_gt, test_data_bl, sample_dir, summary_merged, summary_writer):
        print("[*] Evaluating...")
        psnr_sum = 0
        time_sum = 0
        for idx in range(len(test_data_gt)):
            clean_image = test_data_gt[idx].astype(np.float32)
            bayer_image = test_data_bl[idx].astype(np.float32)
            _, w, h, _  = clean_image.shape
            clean_image = clean_image[:, 0:w//2*2, 0:h//2*2, :]
            bayer_image = bayer_image[:, 0:w//2*2, 0:h//2*2, :]
            val_st_time = time.time()
            #devuelve el output clean
            output_clean_image, noisy_image, psnr_summary = self.sess.run([self.Y, self.X, summary_merged], feed_dict={self.Y_: clean_image, self.X: bayer_image, self.is_training: False})
            val_time = time.time()-val_st_time

            #hace unos clips
            groundtruth = np.clip(clean_image, 0, 255).astype('uint8')
            noisyimage = np.around(np.clip(noisy_image, 0, 255)).astype('uint8')
            #tocar aqui?
            outputimage = np.around(np.clip(output_clean_image, 0, 255)).astype('uint8')
            psnr = imcpsnr(groundtruth, outputimage, 255, 10)
            print("img%d PSNR: %.2f Time: %.2fs" % (idx + 1, psnr, val_time))
            psnr_sum += psnr
            time_sum += val_time
            save_images(os.path.join(sample_dir, 'test%d_%d_%.2f.png' % (idx + 1, iter_num, psnr)), groundtruth, noisyimage, outputimage)
            summary_writer.add_summary(psnr_summary, iter_num)
        avg_psnr = psnr_sum / len(test_data_gt)
        print("--- Validation ---- Average PSNR %.2fdB , Running Time %.2fs---" % (avg_psnr, time_sum))
        avg_psnr_summary = tf.Summary(value=[tf.Summary.Value(tag='Average PSNR', simple_value=avg_psnr)])
        summary_writer.add_summary(avg_psnr_summary, iter_num)

    def train(self, img_labelBatch, img_bayerBatch, eval_data_gt, eval_data_bl, batch_size , ckpt_dir, lr, sample_dir, eval_every_step):
        # load pretrained model
        numStep = len(lr)
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            print("[*] Model restore success!")
        else:
            iter_num = 0
            print("[*] Not find pretrained model!")
        # make summary
        with tf.variable_scope('Loss'):
            tf.summary.scalar('Overall_loss', self.loss)
            tf.summary.scalar('Stage2_lossRGB', self.lossRGB)
        tf.summary.scalar('lr', self.lr)
        #tf.summary.scalar('lr2', self.lr2)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        train_writer = tf.summary.FileWriter('./logs/train', self.sess.graph)
        merged_train = tf.summary.merge_all()
        val_writer = tf.summary.FileWriter('./logs/val')
        val_summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start iter %d : " % (iter_num))
        self.evaluate(iter_num, eval_data_gt, eval_data_bl, sample_dir=sample_dir, summary_merged=val_summary_psnr, summary_writer=val_writer)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)
        try:
            for step in range(iter_num,numStep):
                if coord.should_stop():
                    break
                start_time = time.time()
                label_batch, bayer_batch = self.sess.run([img_labelBatch, img_bayerBatch])#get mini-batch
                _, loss, train_summary, patch = self.sess.run([self.train_op, self.loss, merged_train, self.Y], feed_dict={self.Y_: label_batch, self.X: bayer_batch, self.lr: lr[step], self.is_training: True})
                print("Training: [%4d/%4d] Speed: %4.2fimgs/s, loss: %.6f" % (step + 1, numStep, batch_size/(time.time() - start_time), loss))
                iter_num += 1
                #was set to 1000
                if np.mod(step+1, eval_every_step) == 0:
                    train_writer.add_summary(train_summary, iter_num)
                if np.mod(step + 1, eval_every_step) == 0:# save check points, evaluation
                    self.evaluate(iter_num, eval_data_gt, eval_data_bl, sample_dir=sample_dir, summary_merged=val_summary_psnr, summary_writer=val_writer)
                    self.save(iter_num, ckpt_dir)
        except tf.errors.OutOfRangeError:
            print('epoch limit reached')
            coord.request_stop()
        finally:
            coord.request_stop()
            coord.join(threads)
            train_writer.close()
            val_writer.close()
        print("[*] Finish training.")

    def test(self, test_files_gt, test_files_bl, ckpt_dir, save_dir):
        # init variables
        tf.global_variables_initializer().run()
        assert len(test_files_gt) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")
        for run in range(1): # for accurate running time evaluation, warming-up
            psnr_sum = 0
            psnr_initial_sum = 0
            test_sum = 0
            for idx in range(len(test_files_gt)):
                imagename = os.path.basename(test_files_gt[idx])
                clean_image = load_images(test_files_gt[idx]).astype(np.float32)
                _, w, h, _  = clean_image.shape
                clean_image_crop = clean_image[:, 0:w//2*2, 0:h//2*2, :]
                image_bayer = load_images(test_files_bl[idx]).astype(np.float32)
                image_bayer_crop = image_bayer[:, 0:w//2*2, 0:h//2*2, :]
                test_s_time = time.time()
                output_clean_image = self.sess.run(self.Y, feed_dict={self.Y_: clean_image_crop, self.X: image_bayer_crop, self.is_training: False})
                test_time = time.time()-test_s_time

                # if np.mod(w,4)==1:
                #     output_clean_image = np.pad(output_clean_image, pad_width=((0,0),(0,1),(0,0),(0,0)), mode='symmetric')
                if np.mod(w,2)==1:
                    output_clean_image = np.pad(output_clean_image, pad_width=((0,0),(0,1),(0,0),(0,0)), mode='symmetric')
                # elif np.mod(w,4)==2:
                #     output_clean_image = np.pad(output_clean_image, pad_width=((0,0),(0,2),(0,0),(0,0)), mode='symmetric')
                # elif np.mod(w,4)==3:
                #     output_clean_image = np.pad(output_clean_image, pad_width=((0,0),(0,3),(0,0),(0,0)), mode='symmetric')
                
                # if np.mod(h,4)==1:
                #     output_clean_image = np.pad(output_clean_image, pad_width=((0,0),(0,0),(0,1),(0,0)), mode='symmetric')
                if np.mod(h,2)==1:
                    output_clean_image = np.pad(output_clean_image, pad_width=((0,0),(0,0),(0,1),(0,0)), mode='symmetric')
                # elif np.mod(h,4)==2:
                #     output_clean_image = np.pad(output_clean_image, pad_width=((0,0),(0,0),(0,2),(0,0)), mode='symmetric')
                # elif np.mod(h,4)==3:
                #     output_clean_image = np.pad(output_clean_image, pad_width=((0,0),(0,0),(0,3),(0,0)), mode='symmetric')
                
                
                groundtruth = np.clip(clean_image, 0, 255).astype('uint8')
                noisyimage = np.around(np.clip(image_bayer, 0, 255)).astype('uint8')
                outputimage = np.around(np.clip(output_clean_image, 0, 255)).astype('uint8')
                psnr_bilinear = imcpsnr(groundtruth, noisyimage, 255, 10)
                csnr_bilinear = impsnr(groundtruth, noisyimage, 255, 10)
                psnr = imcpsnr(groundtruth, outputimage, 255, 10)
                csnr = impsnr(groundtruth, outputimage, 255, 10)
                print("Run%d, %s, Bilinear PSNR: %.2fdB, Final PSNR: %.2fdB, Time: %.4fs" % (run, imagename, psnr_bilinear, psnr, test_time))
                print("Groundtruth:")
                print(groundtruth.shape)
                print("Output:")
                print(outputimage.shape)
                psnr_sum += psnr
                psnr_initial_sum += psnr_bilinear
                test_sum += test_time
                save_images(os.path.join(save_dir, imagename), outputimage)
            avg_psnr = psnr_sum / len(test_files_gt)
            avg_psnr_initial = psnr_initial_sum / len(test_files_gt)
            print("--- Test --- Average PSNR Bilinear: %.2fdB, Final: %.2fdB, Running Time: %.4fs ---" % (avg_psnr_initial, avg_psnr, test_sum))

    def self_ensemble_test(self, test_files_gt, test_files_bl, ckpt_dir, save_dir):
        # init variables
        tf.global_variables_initializer().run()
        assert len(test_files_gt) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")
        psnr_sum = 0
        msssim_sum = 0
        ssim_sum = 0
        csnr_sum = np.zeros(3)
        for idx in range(len(test_files_gt)):
            imagename = os.path.basename(test_files_gt[idx])
            clean_image = load_images(test_files_gt[idx]).astype(np.float32)
            _, w, h, _  = clean_image.shape
            clean_image = clean_image[:, 0:w//2*2, 0:h//2*2, :]
            image_bayer = load_images(test_files_bl[idx])[:, 0:w//2*2, 0:h//2*2, :].astype(np.float32)
            image_ensemble = np.zeros([8,image_bayer.shape[1],image_bayer.shape[2],3])
            # mode 1-8
            for mode in range(8):
                if mode == 0:
                    image_bayerRGGB = image_bayer
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.Y_: clean_image, self.X: image_bayerRGGB, self.is_training: False})
                    image_ensemble[mode,:,:,:] = output_clean_image
                    print(imcpsnr(clean_image, output_clean_image, 255, 10))
                elif mode == 1:
                    image_bayer1 = np.flip(image_bayer,2)
                    image_bayerRGGB = image_bayer1
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.Y_: clean_image, self.X: image_bayerRGGB, self.is_training: False})
                    image_ensemble[mode,:,:,:] = np.flip(output_clean_image,2)
                    print(imcpsnr(clean_image, np.flip(output_clean_image,2), 255, 10))
                elif mode == 2:
                    image_bayer1 = np.flip(image_bayer,1)
                    image_bayerRGGB = image_bayer1
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.Y_: clean_image, self.X: image_bayerRGGB, self.is_training: False})
                    image_ensemble[mode,:,:,:] = np.flip(output_clean_image,1)
                    print(imcpsnr(clean_image, np.flip(output_clean_image,1), 255, 10))
                elif mode == 3:
                    image_bayer1 = np.rot90(image_bayer,axes=(1,2))
                    image_bayerRGGB = image_bayer1
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.Y_: clean_image, self.X: image_bayerRGGB, self.is_training: False})
                    image_ensemble[mode,:,:,:] = np.rot90(output_clean_image,3,axes=(1,2))
                    print(imcpsnr(clean_image, np.rot90(output_clean_image,3,axes=(1,2)), 255, 10))
                elif mode == 4:
                    image_bayer1 = np.rot90(image_bayer,2,axes=(1,2))
                    image_bayerRGGB = image_bayer1
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.Y_: clean_image, self.X: image_bayerRGGB, self.is_training: False})
                    image_ensemble[mode,:,:,:] = np.rot90(output_clean_image,2,axes=(1,2))
                    print(imcpsnr(clean_image, np.rot90(output_clean_image,2,axes=(1,2)), 255, 10))
                elif mode == 5:
                    image_bayer1 = np.rot90(image_bayer,3,axes=(1,2))
                    image_bayerRGGB = image_bayer1
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.Y_: clean_image, self.X: image_bayerRGGB, self.is_training: False})
                    image_ensemble[mode,:,:,:] = np.rot90(output_clean_image,axes=(1,2))
                    print(imcpsnr(clean_image, np.rot90(output_clean_image,axes=(1,2)), 255, 10))
                elif mode == 6:
                    image_bayer1 = np.flip(np.rot90(image_bayer,axes=(1,2)),2)
                    image_bayerRGGB = image_bayer1
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.Y_: clean_image, self.X: image_bayerRGGB, self.is_training: False})
                    image_ensemble[mode,:,:,:] = np.rot90(np.flip(output_clean_image,2),3,axes=(1,2))
                    print(imcpsnr(clean_image, np.rot90(np.flip(output_clean_image,2),3,axes=(1,2)), 255, 10))
                elif mode == 7:
                    image_bayer1 = np.flip(np.rot90(image_bayer,3,axes=(1,2)),2)
                    image_bayerRGGB = image_bayer1
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.Y_: clean_image, self.X: image_bayerRGGB, self.is_training: False})
                    image_ensemble[mode,:,:,:] = np.rot90(np.flip(output_clean_image,2),axes=(1,2))
                    print(imcpsnr(clean_image, np.rot90(np.flip(output_clean_image,2),axes=(1,2)), 255, 10))
                else:
                    print('[!]Wrong Mode')
                    exit(0)
            groundtruth = np.clip(clean_image, 0, 255).astype('uint8')
            outputimage = np.average(image_ensemble,axis=0)
            outputimage = np.around(np.clip(outputimage, 0, 255)).astype('uint8')
            outputimage = np.expand_dims(outputimage, 0)
            psnr = imcpsnr(groundtruth, outputimage, 255, 10)
            csnr = impsnr(groundtruth, outputimage, 255, 10)
            msssim = MS_SSIM(groundtruth, outputimage, 10)
            ssim = self.sess.run(SSIM(groundtruth, outputimage, 10)) 
            print("%s, Final PSNR: %.2fdB (R: %.2f, G: %.2f, B: %.2f), MSSSIM: %.5f" % (imagename, psnr, csnr[0], csnr[1], csnr[2], msssim))
            psnr_sum += psnr
            csnr_sum += csnr
            msssim_sum += msssim
            ssim_sum += ssim
            save_images(os.path.join(save_dir, imagename), outputimage)
        avg_psnr = psnr_sum / len(test_files_gt)
        avg_csnr = csnr_sum / len(test_files_gt)
        avg_msssim = msssim_sum / len(test_files_gt)
        avg_ssim = ssim_sum / len(test_files_gt)
        print("--- Test --- Average PSNR Final: %.2fdB (R: %.2f, G: %.2f, B: %.2f), MSSSIM: %.5f, SSIM: %.5f ---" % (avg_psnr, avg_csnr[0], avg_csnr[1], avg_csnr[2], avg_msssim, avg_ssim))

    def save(self, iter_num, ckpt_dir, model_name='CNNCDM-2Stage'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0
