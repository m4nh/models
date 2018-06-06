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
from deeplab.utils.datasets import CityscapesDataset, DatasetUtils
from deeplab.inference import inference


flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')


flags.DEFINE_string('checkpoint_path', None, 'Path to checkpoint')

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


def main(unused_argv):
    print("@"*10, FLAGS.vis_crop_size)
    with tf.Session() as sess:
        model = inference(sess)

        dataset = CityscapesDataset()

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)

        files = sorted(glob.glob(os.path.join("/home/daniele/data/datasets/siteco/DucatiEXP/Images_Ladybug0_0/", "*.jpg")))

        for f in files:
            img = cv2.imread(f)
            # img = colorUtils.centerCrop(img, 1024, 2048)

            height, width = img.shape[:2]
            img = cv2.resize(img, (int(0.5*width), int(0.5*height)), interpolation=cv2.INTER_CUBIC)

            output, logits = model.predict(img)

            outcolor = dataset.buildColorImageNumpy(output, channel_order=DatasetUtils.COLOR_CHANNEL_ORDER_BGR)

            out = np.hstack((img, outcolor))
            cv2.imshow("img", out)
            cv2.waitKey(0)


if __name__ == '__main__':
    tf.app.run()
