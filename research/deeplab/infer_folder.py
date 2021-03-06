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

flags.DEFINE_multi_integer('atrous_rates', [6, 12, 18],
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

flags.DEFINE_integer('num_classes', 19,
                     'Number of classes used for training')

flags.DEFINE_multi_integer('vis_crop_size', [2448, 2048],
                           'Crop size [height, width] for visualization.')


def rescaleImage(img, rescale):
    if len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    h = int(rescale*h)
    w = int(rescale*w)
    newimg = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return newimg, h, w


def main(unused_argv):

    with tf.Session() as sess:

        dataset = CityscapesDataset()

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)

        rescale = 0.5
        input_path = "/home/daniele/data/datasets/siteco/DucatiEXP/Images_Ladybug0_5/"
        output_path = "/home/daniele/data/datasets/siteco/DucatiEXP/Segmentations/Images_Ladybug0_5_reduced/"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        files = sorted(glob.glob(os.path.join(input_path, "*.jpg")))

        sample_image = cv2.imread(files[0])

        sample_image, h, w = rescaleImage(sample_image, rescale)

        #flags.vis_crop_size = [h, w]
        # print(sample_image.shape)
        # import sys
        # sys.exit(0)

        model = inference(sess, crop_size=[h, w])

        for f in files:
            img = cv2.imread(f)
            # img = colorUtils.centerCrop(img, 1024, 2048)

            img, h, w = rescaleImage(img, rescale)
            # height, width = img.shape[:2]
            # img = cv2.resize(img, (int(0.5*width), int(0.5*height)), interpolation=cv2.INTER_CUBIC)

            output, logits = model.predict(img)

            outcolor = dataset.buildColorImageNumpy(output, channel_order=DatasetUtils.COLOR_CHANNEL_ORDER_BGR)

            out = np.hstack((img, outcolor))
            cv2.imshow("img", out)
            cv2.waitKey(1)
            print("LOGIT SHAPES: ", logits.shape)

            outimg, h, w = rescaleImage(outcolor, 1/rescale)
            output_r, h, w = rescaleImage(output, 1/rescale)

            image_name = os.path.splitext(os.path.basename(f))[0]
            colored_image_name = image_name+"_colorsegmentation.jpg"
            labels_image_name = image_name+"_segmentation.png"
            logits_name = image_name+"_logits.npy"

            cv2.imwrite(os.path.join(output_path, colored_image_name), outimg)
            cv2.imwrite(os.path.join(output_path, labels_image_name), output_r)
            #np.save(os.path.join(output_path, logits_name), logits)


if __name__ == '__main__':
    tf.app.run()
