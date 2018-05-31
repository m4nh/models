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
        semantic_predictions_value = self.sess.run(self.semantic_predictions, feed_dict={
                                                   self.image_placeholder: input_npy})
        print("Time needed: {:.2f}s".format(time.time()-start))
        if save_path != None:
            save_annotation.save_annotation(semantic_predictions_value, os.path.dirname(save_path),
                                            os.path.splitext(os.path.basename(save_path))[0], add_colormap=False)
        return semantic_predictions_value

    def build_model(self):
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


def main(unused_argv):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    #### load npy rgb ####
    #img = np.load("/home/pier/Desktop/rgb.npy")
    img = cv2.imread(
        "/home/daniele/data/datasets/siteco/DucatiEXP/Images_Ladybug0_0/Frame_F_0.jpg")
    img2 = np.zeros(img.shape, np.uint8)
    img2[:, :, 0] = img[:, :, 2]
    img2[:, :, 1] = img[:, :, 1]
    img2[:, :, 2] = img[:, :, 0]
    cv2.imshow("img", img2)
    cv2.waitKey(0)
    with tf.Session() as sess:
        model = inference(sess)
        #### optional save_path, full path to output ####
        output = model.predict(img2)
        output2 = np.zeros(img.shape, np.uint8)
        output2[:, :, 0] = output
        output2[:, :, 1] = output
        output2[:, :, 2] = output
        print(output.shape, np.min(output), np.max(output))
        cv2.imshow("img", output2)
        cv2.waitKey(0)


if __name__ == '__main__':
    tf.app.run()
