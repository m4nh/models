import math
import os.path
import time
import glob
import cv2
import numpy as np
from collections import namedtuple
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import save_annotation
from deeplab.utils import datasets
import colorUtils
from deeplab.utils.datasets import CityscapeDataset, DatasetUtils

Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).

    'trainId',  # An integer ID that overwrites the ID above, when creating ground truth
    # images for training.
    # For training, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('road',       7,        0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk',   8,        1, 'flat', 1, False, False, (244, 35, 232)),
    Label('building',   11,        2, 'construction',
          2, False, False, (70, 70, 70)),
    Label('wall', 12,        3, 'construction',
          2, False, False, (102, 102, 156)),
    Label('fence', 13,        4, 'construction',
          2, False, False, (190, 153, 153)),
    Label('pole', 17,        5, 'object', 3, False, False, (153, 153, 153)),
    Label('traffic light', 19,        6, 'object',
          3, False, False, (250, 170, 30)),
    Label('traffic sign', 20,        7, 'object',
          3, False, False, (220, 220,  0)),
    Label('vegetation', 21,        8, 'nature',
          4, False, False, (107, 142, 35)),
    Label('terrain', 22,        9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23,       10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24,       11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25,       12, 'human', 6, True, False, (255,  0,  0)),
    Label('car', 26,       13, 'vehicle', 7, True, False, (0,  0, 142)),
    Label('truck', 27,       14, 'vehicle', 7, True, False, (0,  0, 70)),
    Label('bus', 28,       15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('train', 31,       16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32,       17, 'vehicle', 7, True, False, (0,  0, 230)),
    Label('bicycle', 33,       18, 'vehicle', 7, True, False, (119, 11, 32)),
]


slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('checkpoint_path', None, 'Path to checkpoint')

# Settings for visualizing the model.

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.

flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

flags.DEFINE_integer('num_classes', 19,
                     'Number of classes used for training')

flags.DEFINE_multi_integer('vis_crop_size', [2448, 2048],
                           'Crop size [height, width] for visualization.')


class inference(object):
    def __init__(self, sess):
        self.sess = sess
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
            crop_size=FLAGS.vis_crop_size,
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
            crop_size=FLAGS.vis_crop_size,
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


def main(unused_argv):

    with tf.Session() as sess:
        model = inference(sess)

        dataset = CityscapeDataset()

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)

        files = sorted(glob.glob(os.path.join("/home/daniele/data/datasets/siteco/DucatiEXP/Images_Ladybug0_0/", "*.jpg")))

        for f in files:
            img = cv2.imread(f)
            img = colorUtils.centerCrop(img, 1024, 2048)

            height, width = img.shape[:2]
            img = cv2.resize(img, (int(0.5*width), int(0.5*height)), interpolation=cv2.INTER_CUBIC)

            output, logits = model.predict(img)

            outcolor = dataset.buildColorImageNumpy(output, channel_order=DatasetUtils.COLOR_CHANNEL_ORDER_BGR)

            out = np.hstack((img, outcolor))
            cv2.imshow("img", out)
            cv2.waitKey(0)


if __name__ == '__main__':
    tf.app.run()
