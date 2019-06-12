import tensorflow as tf

# network structure
def subpixel_LR2HR(image_LR):
    c = image_LR.shape[3]
    if c==4:#64*50*50*4
        img_HR = tf.depth_to_space(image_LR, 2)
        return img_HR
    elif c==8:#64*50*50*8
        img_HR1 = tf.depth_to_space(image_LR[:,:,:,0:4], 2)
        img_HR2 = tf.depth_to_space(image_LR[:,:,:,4:8], 2)
        return tf.concat([img_HR1, img_HR2], 3)
    elif c==12:#128*12*12*12
        img_HR1 = tf.depth_to_space(image_LR[:,:,:,0:4], 2)
        img_HR2 = tf.depth_to_space(image_LR[:,:,:,4:8], 2)
        img_HR3 = tf.depth_to_space(image_LR[:,:,:,8:12], 2)
        return tf.concat([img_HR1, img_HR2, img_HR3], 3)
    else:
        print('ERROR!')

def subpixel_HR2LR_new(image_HR):
    c = image_HR.shape[3]
    if c==1:#64*50*50*4
        img_LR = tf.space_to_depth(image_HR, 2)
        return img_LR
    elif c==2:#64*50*50*8
        img_LR1 = tf.space_to_depth(image_HR[:,:,:,0:1], 2)
        img_LR2 = tf.space_to_depth(image_HR[:,:,:,1:2], 2)
        return tf.concat([img_LR1, img_LR2], 3)
    elif c==3:#128*24*24*12
        img_LR1 = tf.space_to_depth(image_HR[:,:,:,0:1], 2)
        img_LR2 = tf.space_to_depth(image_HR[:,:,:,1:2], 2)
        img_LR3 = tf.space_to_depth(image_HR[:,:,:,2:3], 2)
        return tf.concat([img_LR1, img_LR2, img_LR3], 3)
    else:
        print('ERROR!')

def parametric_relu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32,trainable=True)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg

def subpixel_new(input, is_training=True):# original structure with regularizer
    #64(9)-32(1)-32(7)-64(1)-1[9]-s2
    #number of filters
    n1=64
    n2=32
    n3=32
    n4=64
    n5=1
    #filters spatial size
    f1=9
    f2=1
    f3=7
    f4=1
    f5=9
    #stride conv and deconv
    stride=2

    #My_initial = tf.contrib.layers.variance_scaling_initializer()
    My_initial = tf.initializers.truncated_normal(stddev=0.001)
    My_regular = None
    #same as ARCNN but with stride
    with tf.variable_scope('Pre_processing'):
        R4G4B4 = subpixel_HR2LR_new(input)
    with tf.variable_scope('First_layers'):
        with tf.variable_scope('1-Feature_extraction'):
            output = tf.layers.conv2d(R4G4B4, n1, f1, strides=stride, padding='same', activation=None, kernel_initializer=My_initial, kernel_regularizer=My_regular)
            output = parametric_relu(output)
        
        #no padding as 32 1x1 (valid)
        with tf.variable_scope('2-Shrinking'):
            output = tf.layers.conv2d(output, n2, f2, padding='valid', activation=None, kernel_initializer=My_initial, kernel_regularizer=My_regular)
            output = parametric_relu(output)
        #zero padding
        with tf.variable_scope('3-Enhancement'):
            output = tf.layers.conv2d(output, n3, f3, padding='same', kernel_initializer=My_initial, kernel_regularizer=My_regular)
            output = parametric_relu(output)

        with tf.variable_scope('4-Mapping'):
            output = tf.layers.conv2d(output, n4, f4, padding='valid', kernel_initializer=My_initial, kernel_regularizer=My_regular)
        #Deconvolution
        # 12 filters as we have to do the LR to HR
    with tf.variable_scope('Remaining_layers'):
        with tf.variable_scope('5-Reconstruction'):
            output = tf.layers.conv2d_transpose(output, 12, f5, strides=stride, padding='same', kernel_initializer=My_initial, kernel_regularizer=My_regular)
    return subpixel_LR2HR(output)