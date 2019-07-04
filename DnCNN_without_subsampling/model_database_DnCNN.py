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
    elif c==12:#64*50*50*12
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
    elif c==3:#64*50*50*12
        img_LR1 = tf.space_to_depth(image_HR[:,:,:,0:1], 2)
        img_LR2 = tf.space_to_depth(image_HR[:,:,:,1:2], 2)
        img_LR3 = tf.space_to_depth(image_HR[:,:,:,2:3], 2)
        return tf.concat([img_LR1, img_LR2, img_LR3], 3)
    else:
        print('ERROR!')

def subpixel_new(input, is_training=True):# original structure with regularizer
    num_feature = 64
    #Delving deep into rectifiers initialization
    My_initial = tf.contrib.layers.variance_scaling_initializer()
    My_regular = tf.contrib.layers.l2_regularizer(scale=0.0001)
    #My_regular=None
    # with tf.variable_scope('Pre_processing'):
    #     R4G4B4 = subpixel_HR2LR_new(input)
    R4G4B4=input
    with tf.variable_scope('stage1'):
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(R4G4B4, num_feature, 3, padding='same', activation=tf.nn.relu, kernel_initializer=My_initial, kernel_regularizer=My_regular)
        for layers in range(2, 19 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', kernel_initializer=My_initial, kernel_regularizer=My_regular, name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, moving_variance_initializer=tf.zeros_initializer(), epsilon=1e-5, training=is_training))
        with tf.variable_scope('block20'):
            output = tf.layers.conv2d(output,3, 3, padding='same', kernel_initializer=My_initial, kernel_regularizer=My_regular)#batch*50*50*12
            S1InterRGB = R4G4B4 + output
    return S1InterRGB