import time
import os

import utils.realtime_renderer as rtr

from utils.tf_utils import *


class mv3d():

    def __init__(self, sess):

        self.sess = sess
        self.batch_size = 64
        self.input_shape = [128, 128, 3]
        self.output_shape = [128, 128, 4]
        self.max_iter = 1000000
        self.start_iter = 0

        self.log_folder = "logs/bg_nodm"
        self.train_samples_folder = "samples/bg_nodm/train"
        self.test_samples_folder = "samples/bg_nodm/test"
        self.snapshots_folder = "snapshots/bg_nodm"

        self.rend = rtr.RealTimeRenderer(self.batch_size)
        self.rend.load_model_names("data/cars_training.txt")

        self.test_images1, self.test_images2,\
            self.test_dm2, self.test_labels = load_test_set(True, True, 100.0)

    def buildModel(self):
        self.images1 = tf.placeholder(tf.float32,
                                      [self.batch_size] + self.input_shape,
                                      name='input_images')
        self.images2 = tf.placeholder(tf.float32,
                                      [self.batch_size] + self.output_shape,
                                      name='gt_images')
        self.labels = tf.placeholder(tf.float32, [self.batch_size, 5],
                                     name='labels')

        # convolutional encoder
        e0 = lrelu(conv2d_msra(self.images1, 16, 5, 5, 2, 2, "e0"))
        e0_0 = lrelu(conv2d_msra(e0, 32, 3, 3, 1, 1, "e0_1"))
        e1 = lrelu(conv2d_msra(e0_0, 32, 5, 5, 2, 2, "e1"))
        e1_0 = lrelu(conv2d_msra(e1, 32, 3, 3, 1, 1, "e1_1"))
        e2 = lrelu(conv2d_msra(e1_0, 64, 3, 3, 2, 2, "e2"))
        e2_0 = lrelu(conv2d_msra(e2, 64, 3, 3, 1, 1, "e2_1"))
        e3 = lrelu(conv2d_msra(e2_0, 128, 3, 3, 2, 2, "e3"))
        e3_0 = lrelu(conv2d_msra(e3, 128, 3, 3, 1, 1, "e3_1"))
        e4 = lrelu(conv2d_msra(e3_0, 256, 3, 3, 2, 2, "e4"))
        e4_0 = lrelu(conv2d_msra(e4, 256, 3, 3, 1, 1, "e4_1"))
        e4r = tf.reshape(e4_0, [self.batch_size, 4096])
        e5 = lrelu(linear_msra(e4r, 4096, "fc1"))

        # angle processing
        a0 = lrelu(linear_msra(self.labels, 64, "a0"))
        a1 = lrelu(linear_msra(a0, 64, "a1"))
        a2 = lrelu(linear_msra(a1, 64, "a2"))

        concated = tf.concat(1, [e5, a2])

        # joint processing
        a3 = lrelu(linear_msra(concated, 4096, "a3"))
        a4 = lrelu(linear_msra(a3, 4096, "a4"))
        a5 = lrelu(linear_msra(a4, 4096, "a5"))
        a5r = tf.reshape(a5, [self.batch_size, 4, 4, 256])

        # convolutional decoder
        d4 = lrelu(deconv2d_msra(a5r, [self.batch_size, 8, 8, 128],
                                 3, 3, 2, 2, "d4"))
        d4_0 = lrelu(conv2d_msra(d4, 128, 3, 3, 1, 1, "d4_1"))
        d3 = lrelu(deconv2d_msra(d4_0, [self.batch_size, 16, 16, 64],
                                 3, 3, 2, 2, "d3"))
        d3_0 = lrelu(conv2d_msra(d3, 64, 3, 3, 1, 1, "d3_1"))
        d2 = lrelu(deconv2d_msra(d3_0, [self.batch_size, 32, 32, 32],
                                 5, 5, 2, 2, "d2"))
        d2_0 = lrelu(conv2d_msra(d2, 32, 3, 3, 1, 1, "d2_1"))
        d1 = lrelu(deconv2d_msra(d2_0, [self.batch_size, 64, 64, 32],
                                 5, 5, 2, 2, "d1"))
        d1_0 = lrelu(conv2d_msra(d1, 32, 3, 3, 1, 1, "d1_1"))
        d0 = lrelu(deconv2d_msra(d1_0, [self.batch_size, 128, 128, 16],
                                 5, 5, 2, 2, "d0"))
        self.gen = tf.nn.tanh(conv2d_msra(d0, 4, 3, 3, 1, 1, "d0_1"))

        gt_cm = tf.slice(self.images2, [0, 0, 0, 0], [64, 128, 128, 3])
        gt_sm = tf.slice(self.images2, [0, 0, 0, 3], [64, 128, 128, 1])
        sm = gt_sm
        gt_sm = gt_sm * 0.75
        pr_cm = tf.slice(self.gen, [0, 0, 0, 0], [64, 128, 128, 3])
        pr_sm = tf.slice(self.gen, [0, 0, 0, 3], [64, 128, 128, 1])

        self.loss = euclidean_loss(tf.mul(gt_cm, sm), tf.mul(pr_cm, sm)) +\
            0.1 * euclidean_loss(gt_sm, pr_sm)

        self.training_summ = tf.scalar_summary("training_loss", self.loss)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(max_to_keep=20)

    def restore(self):
        self.start_iter = load_snapshot(self.saver, self.sess,
                                        self.snapshots_folder)

    def generate_sample_set(self, path, im1, im2, gen, iter_num):

        pr_cm = tf.slice(gen, [0, 0, 0, 0], [self.batch_size, 128, 128, 3])
        pr_sm = tf.slice(gen, [0, 0, 0, 3], [self.batch_size, 128, 128, 1])

        save_images(tf.mul(pr_cm, pr_sm*4.0/3.0).eval(session=self.sess),
                    [8, 8], path + "/output_%s.png" % (iter_num))
        save_images(gen[:, :, :, 3], [8, 8],
                    path + "/sm_%s.png" % (iter_num), False)
        save_images(np.array(im2)[:, :, :, 0:3], [8, 8],
                    path + '/tr_gt_%s.png' % (iter_num))
        save_images(np.array(im2)[:, :, :, 3], [8, 8],
                    path + '/dm_gt_%s.png' % (iter_num), False)
        save_images(np.array(im1), [8, 8],
                    path + '/tr_input_%s.png' % (iter_num))

    def test(self, global_iter):
        self.test_iter = 19
        sm_path = os.path.join(self.test_samples_folder,
                               str(global_iter).zfill(8))
        if not os.path.exists(sm_path):
            os.mkdir(sm_path)
        local_loss = 0.0
        for i in range(0, self.test_iter):
            batch_images1 = self.test_images1[i*self.batch_size:
                                              (i+1)*self.batch_size]
            cl_im = np.asarray(self.test_images2[i*self.batch_size:
                               (i+1)*self.batch_size])
            dm_im = np.asarray(self.test_dm2[i*self.batch_size:
                               (i+1)*self.batch_size]).reshape(
                               (self.batch_size, 128, 128, 1))
            batch_images2 = np.concatenate((cl_im, dm_im), axis=3)
            batch_labels = self.test_labels[i*self.batch_size:
                                            (i+1)*self.batch_size]
            output = self.sess.run([self.gen, self.loss],
                                   feed_dict={
                                        self.images1: batch_images1,
                                        self.images2: batch_images2,
                                        self.labels: batch_labels})

            self.generate_sample_set(sm_path, batch_images1,
                                     batch_images2, output[0], i)
            local_loss += float(output[1])

        total_loss = local_loss / self.test_iter
        print("[i: %s] [test loss: %.6f]" %
              (global_iter, total_loss))
        if self.writer is not None:
            log_value(self.writer, total_loss, 'test_loss', global_iter)

    def train(self):
        optim = tf.train.AdamOptimizer(
            0.0001, beta1=0.9).minimize(
            self.loss, var_list=self.t_vars)

        self.writer = tf.train.SummaryWriter(
            self.log_folder, self.sess.graph_def)
        tf.global_variables_initializer().run()
        self.restore()

        for i in range(self.start_iter, self.max_iter):

            iteration_start_time = time.time()
            cl1, dm1, cl2, dm2, lb = self.rend.render(10.0)
            mask = (dm2 < 0.7)

            full_output = np.concatenate(
                (cl2, mask.reshape(self.batch_size, 128, 128, 1)), axis=3)

            output = self.sess.run(
                [optim, self.loss, self.training_summ],
                feed_dict={self.images1: cl1,
                           self.images2: full_output,
                           self.labels: lb})

            if np.mod(i, 400) == 0:
                self.test(i)
                sm_path = os.path.join(self.train_samples_folder,
                                       str(i).zfill(8))
                if not os.path.exists(sm_path):
                    os.mkdir(sm_path)
                sm = self.sess.run(self.gen,
                                   feed_dict={self.images1: cl1,
                                              self.images2: full_output,
                                              self.labels: lb})
                self.generate_sample_set(sm_path, cl1,
                                         full_output, sm, i)

            if np.mod(i, 5000) == 1:
                save_snapshot(self.saver, self.sess,
                              self.snapshots_folder, i)

            if np.mod(i, 10) == 0:
                summ_str = output[2]
                self.writer.add_summary(summ_str, i)

            print("[i: %s] [time: %s] [global time: %s] [train loss: %.6f]" %
                  (i, time.time() - iteration_start_time,
                   time.time() - global_start_time, output[1]))

global_start_time = time.time()

with tf.Session() as sess:
    net = mv3d(sess)
    net.buildModel()

    # ---TEST---
    net.restore()
    net.test(0)

    # ---TRAIN---
    # net.train()
