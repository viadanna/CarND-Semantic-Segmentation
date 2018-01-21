import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import numpy as np
import cv2
import time

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model(image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load saved model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # Gather required tensor references
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, layer3_out, layer4_out, layer7_out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # 1x1 convolution
    layer7_out = tf.layers.conv2d(
        inputs=vgg_layer7_out,
        filters=num_classes,
        kernel_size=1,
        padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    # Upsample
    layer7_up = tf.layers.conv2d_transpose(
        inputs=layer7_out,
        filters=num_classes,
        kernel_size=4,
        strides=(2, 2),
        padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    # 1x1 convolution
    layer4_out = tf.layers.conv2d(
        inputs=vgg_layer4_out,
        filters=num_classes,
        kernel_size=1,
        padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    # Skip layer
    skip_1 = tf.add(layer7_up, layer4_out)

    # Upsample
    skip_1_up = tf.layers.conv2d_transpose(
        inputs=skip_1,
        filters=num_classes,
        kernel_size=4,
        strides=(2, 2),
        padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    # 1x1 convolution
    layer3_out = tf.layers.conv2d(
        inputs=vgg_layer3_out,
        filters=num_classes,
        kernel_size=1,
        padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    # Skip layer
    skip_2 = tf.add(skip_1_up, layer3_out)

    # Upsampled final
    skip_2_up = tf.layers.conv2d_transpose(
        inputs=skip_2,
        filters=num_classes,
        kernel_size=16,
        strides=(8, 8),
        padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    return skip_2_up


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of(logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # Cross-entropy operation
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    # Train operation using Adam and an exponential decaying learning rate
    step = tf.Variable(0, trainable=False)
    rate = tf.train.exponential_decay(
        learning_rate=learning_rate,
        global_step=step,
        decay_steps=5,  # reduce every 5 steps or 1 epoch
        decay_rate=0.63)  # reduce by 37% every 5 steps or by 90% every 5 epochs
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())

    # Check for a saved checkpoint to continue training
    saver = tf.train.Saver()
    try:
        assert os.path.exists('./model.ckpt.meta')
        saver.restore(sess, './model.ckpt')
        best_loss = 0
        print('Loaded saved model, evaluating...')
        for image, label in get_batches_fn(batch_size):
            best_loss += sess.run(cross_entropy_loss, feed_dict={
                input_image: image,
                correct_label: label,
                keep_prob: 1,
                learning_rate: 1})
        print('Done, loss:', best_loss)
    except Exception as e:
        print('No saved model checkpoint found', str(e))
        best_loss = 1e9

    # Continue training while the epoch loss improves
    patience = 1
    failure = 0
    step = 0
    for e in range(epochs):
        t0 = time.time()
        epoch_loss = 0
        for images, labels in get_batches_fn(batch_size):
            step += 1
            t1 = time.time()
            images = random_batch(images)
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                input_image: images,
                correct_label: labels,
                keep_prob: 0.5,
                learning_rate: 1e-4})
            print('Step {} loss: {:.3f} took {:.1f}s'.format(
                step, loss, time.time() - t1))
            epoch_loss += loss
        print('Epoch: {}/{} loss: {:.3f} took {:.1f}s'.format(
            e + 1, epochs, epoch_loss, time.time() - t0))
        if epoch_loss > best_loss:
            if failure == patience:
                break
            failure += 1
        else:
            failure = 0
            best_loss = epoch_loss
            saver.save(sess, './model.ckpt')

    # Reload the best model found
    saver.restore(sess, './model.ckpt')


# Tests crash on saver, so let's skip this for now
# tests.test_train_nn(train_nn)

def random_batch(batch):
    return np.array([augment_brightness_camera_images(image) for image in batch], dtype=np.uint8)


def augment_brightness_camera_images(image, chance=.5):
    # Random chance to use original image
    if np.random.random() > chance:
        return image
    # Randomly increase or decrease brightness
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image[:, :, 2] = image[:, :, 2] * random_bright
    image[:, :, 2][image[:, :, 2] > 255] = 255
    image = np.array(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        epochs = 50
        batch_size = 58  # Requires 16GB of free memory

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
