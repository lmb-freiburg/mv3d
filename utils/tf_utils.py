import tensorflow as tf
import math
import numpy as np
import scipy
import scipy.misc
import time
import os
import re


def log_value(writer, value, tag, step):
    s = tf.Summary()
    s.value.extend([tf.Summary.Value(tag=tag, simple_value=value)])
    ev = tf.Event(wall_time=time.time(), step=int(step), summary=s)
    writer.add_event(ev)


def euclidean_loss(input1, input2):
    return tf.reduce_mean(tf.reduce_sum(tf.pow(tf.sub(input1, input2), 2), 3))


def l1_loss(input1, input2):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(tf.sub(input1, input2)), 3))


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def linear_msra(input_, output_size, name):
    msra_coeff = 1.0
    shape = input_.get_shape().as_list()
    fan_in = int(input_.get_shape()[-1])
    stddev = msra_coeff * math.sqrt(2. / float(fan_in))

    with tf.variable_scope(name):
        matrix = tf.get_variable(
          "Matrix", [shape[1], output_size], tf.float32,
          tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable(
          'b', [output_size, ], initializer=tf.constant_initializer(value=0.))

    return tf.matmul(input_, matrix) + b


def conv2d_msra(input_, output_dim, k_h, k_w, d_h, d_w, name):
    with tf.variable_scope(name):
        msra_coeff = 1.0
        fan_in = k_h * k_w * int(input_.get_shape()[-1])
        stddev = msra_coeff * math.sqrt(2. / float(fan_in))
        w = tf.get_variable(
            'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable(
            'b', [output_dim, ],
            initializer=tf.constant_initializer(value=0.0))
        conv = tf.nn.conv2d(
            input_, w, strides=[1, d_h, d_w, 1], padding='SAME') + b

    return conv


def deconv2d_msra(input_, output_shape, k_h, k_w, d_h, d_w, name):
    with tf.variable_scope(name):
        msra_coeff = 1.0
        fan_in = k_h * k_w * int(input_.get_shape()[-1])
        stddev = msra_coeff *\
            math.sqrt(2.0 / float(fan_in) * float(d_h) * float(d_w))
        w = tf.get_variable(
            'w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(
            input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
    return deconv


def save_images(images, size, image_path, color=True):
    h, w = images.shape[1], images.shape[2]
    if color is True:
        img = np.zeros((h * size[0], w * size[1], 3))
    else:
        img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = math.floor(idx / size[1])
        if color is True:
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        else:
            img[j*h:j*h+h, i*w:i*w+w] = image
    if color is True:
        scipy.misc.toimage(rescale_image(img),
                           cmin=0, cmax=255).save(image_path)
    else:
        scipy.misc.toimage(rescale_dm(img), cmin=0, cmax=65535,
                           low=0, high=65535, mode='I').save(image_path)


def save_movie_frames(images, image_path, offset=0, dm=False):
    for idx, image in enumerate(images):
        if dm is True:
            scipy.misc.toimage(
                rescale_dm(image), cmin=0, cmax=65535,
                low=0, high=65535, mode='I').save(
                image_path + "/dm_" + str(idx+offset) + ".png")
        else:
            scipy.misc.toimage(
                rescale_image(image), cmin=0, cmax=255).save(
                    image_path + "/cl_" + str(idx+offset) + ".png")


def rescale_image(image):
    new_im = (image / 1.5 + 0.5) * 255
    return new_im


def rescale_dm(image):
    new_im = (image / 1.5 + 0.5) * 65535
    return new_im


def load_test_set(normal, bg, rad_factor=1000.0):
    print("loading test set...")
    script_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_path, '../data')

    test_images1 = []
    test_images2 = []
    test_dm2 = []
    test_labels = []

    if normal is True:
        path = os.path.join(path, "test_normal_rendered")
    else:
        path = os.path.join(path, "test_difficult_rendered")

    if bg is True:
        path = os.path.join(path, "bg")
    else:
        path = os.path.join(path, "nobg")

    for dirname, dirnames, filenames in os.walk(path):
        model_name = dirname.split('/')
        if len(model_name) > 4:
            model_name = model_name[ len(model_name)-1 ]
            print(model_name)
            for filename in filenames:
                im = scipy.misc.imread(os.path.join(dirname, filename))
                parts = filename.split(".")[0].split("_")
                if parts[1] == "cl":
                    im = (np.array(im) / 255.0 - 0.5) * 1.5
                    if int(parts[2]) == 0:
                        test_images1.append(im)
                    else:
                        test_images2.append(im)
                        rad = float(parts[3]) / rad_factor
                        e1 = math.sin(math.radians(float(parts[5])))
                        e2 = math.cos(math.radians(float(parts[5])))
                        a1 = math.sin(math.radians(float(parts[4])))
                        a2 = math.cos(math.radians(float(parts[4])))
                        test_labels.append(
                            np.array([rad, e1, e2, a1, a2]))
                else:
                    im = (np.array(im) / 65535.0 - 0.5) * 1.5
                    if int(parts[2]) == 1:
                        test_dm2.append(im)
    print("done.")
    return test_images1, test_images2, test_dm2, test_labels


def save_snapshot(saver, session, path, step):
    saver.save(session, os.path.join(path,
               "snapshot" + str(step)), global_step=step)


def load_snapshot(saver, session, path):
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt is not None:
        print("loading " + ckpt.model_checkpoint_path + "...")
        saver.restore(session, ckpt.model_checkpoint_path)
        num_iter = int(re.match('.*-(\d*)$',
                       ckpt.model_checkpoint_path).group(1))
        print("done.")
        return num_iter
