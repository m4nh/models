import math
import os.path
import time
import cv2
import numpy as np
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import save_annotation

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS


class inference(object):
    def __init__(self, sess, crop_size):
        self.sess = sess
        self.crop_size = crop_size
        self.build_model()
        init_op = [tf.local_variables_initializer(
        ), tf.global_variables_initializer()]
        tf.train.get_or_create_global_step()
        last_checkpoint = FLAGS.checkpoint_path
        self.sess.run(init_op)
        saver = tf.train.Saver(slim.get_variables_to_restore())
        saver.restore(self.sess, last_checkpoint)

    def predict(self, input_npy, save_path=None):
        start = time.time()
        semantic_predictions_value, logits_predictions_value = self.sess.run(
            [self.semantic_predictions, self.logits],
            feed_dict={self.image_placeholder: input_npy}
        )
        print("Time needed: {:.2f}s".format(time.time()-start))
        if save_path != None:
            save_annotation.save_annotation(semantic_predictions_value, os.path.dirname(save_path),
                                            os.path.splitext(os.path.basename(save_path))[0], add_colormap=False)

        return semantic_predictions_value, logits_predictions_value

    def build_model_simple(self):
        self.image_placeholder = tf.placeholder(
            tf.uint8, shape=[None, None, 3])
        image = tf.expand_dims(tf.to_float(self.image_placeholder), axis=0)
        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: FLAGS.num_classes},
            crop_size=self.crop_size,
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)
        predictions = model.predict_labels(
            image,
            model_options=model_options,
            image_pyramid=FLAGS.image_pyramid)
        self.semantic_predictions = tf.squeeze(predictions[common.OUTPUT_TYPE])

    def build_model(self):
        self.image_placeholder = tf.placeholder(
            tf.uint8, shape=[None, None, 3])
        image = tf.expand_dims(tf.to_float(self.image_placeholder), axis=0)
        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: FLAGS.num_classes},
            crop_size=self.crop_size,
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)
        self.softmax = model.predict_logits(
            image,
            model_options=model_options,
            image_pyramid=FLAGS.image_pyramid)

        self.semantic_predictions = tf.squeeze(
            tf.argmax(self.softmax[common.OUTPUT_TYPE], axis=-1)
        )
        self.logits = tf.squeeze(self.softmax[common.OUTPUT_TYPE])
