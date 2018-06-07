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
import postcard

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


#################################
# Globals
#################################
rescale = 0.5
dataset = CityscapesDataset()
inference_network = None


def semanticCallback(header, input_image):
    # global inference_model, image_placeholder, session
    # print(inference_model, session)
    # if inference_model is not None and session is not None:
    global inference_network, rescale

    exp_h, exp_w = inference_network.crop_size

    img, h, w = rescaleImage(input_image, rescale)

    if exp_h != h or exp_w != w:
        return "error.BAD_RESOLUTION", None

    output, logits = inference_network.predict(img)

    if header.command == "segmentation_deeplab":
        output = output.astype(np.uint8)
        output_rescaled, h, w = rescaleImage(output, 1/rescale)
        return "ok", output_rescaled
    elif header.command == "segmentation_deeplab_color":
        outcolor = dataset.buildColorImageNumpy(output, channel_order=DatasetUtils.COLOR_CHANNEL_ORDER_BGR)
        outcolor_rescaled, h, w = rescaleImage(outcolor, 1/rescale)
        return "ok", outcolor_rescaled
    else:
        return "error.INVALID_COMMAND", None

    return "error.DEEP_ERROR", None


socket = postcard.PostcardServer.AcceptingSocket('0.0.0.0', 8000)
print("Server Running...")


def main(unused_argv):
    global inference_network
    with tf.Session() as sess:

        input_path = "/home/daniele/data/datasets/siteco/DucatiEXP/Images_Ladybug0_0/"
        output_path = "/home/daniele/data/datasets/siteco/DucatiEXP/Segmentations/Images_Ladybug0_0_reduced/"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        files = sorted(glob.glob(os.path.join(input_path, "*.jpg")))

        sample_image = cv2.imread(files[0])

        h, w = FLAGS.vis_crop_size
        h = int(rescale*h)
        w = int(rescale*w)

        #flags.vis_crop_size = [h, w]
        # print(sample_image.shape)
        # import sys
        # sys.exit(0)

        inference_network = inference(sess, crop_size=[h, w])

        while True:
            print("Waiting for connection....")
            connection, address = socket.accept()
            server = postcard.PostcardServer(
                connection, address, data_callback=semanticCallback)


if __name__ == '__main__':
    tf.app.run()
