import numpy as np


class DatasetUtils(object):
    COLOR_CHANNEL_ORDER_RGB = [0, 1, 2]
    COLOR_CHANNEL_ORDER_BGR = [2, 1, 0]


class GenericDataset(object):

    def __init__(self, dataset_name=""):
        self.dataset_name = dataset_name
        self.colormap = []
        self.names = []

    def size(self):
        return len(self.colormap)

    def buildColorImageNumpy(self, pred_sem, channel_order=[0, 1, 2]):
        """ Builds a Color Image from Segmentation Output managing Numpy matrices"""
        max_label = len(self.names)
        p = np.stack([pred_sem, pred_sem, pred_sem], axis=-1)
        m = np.zeros_like(p)
        for i in range(max_label):
            mi = np.multiply(np.ones_like(p), self.colormap[i])
            m = np.where(np.equal(p, i), mi, m)
        m = m.astype(np.uint8)
        m = m[:, :, channel_order]
        return m

    def buildColorImageTf(self, pred_sem):
        """ Builds a Color Image from Segmentation Output managing Tensroflow Tensors"""
        max_label = len(self.names)
        p = tf.squeeze(tf.cast(pred_sem, tf.uint8), axis=-1)
        p = tf.stack([p, p, p], axis=-1)
        m = tf.zeros_like(p)
        for i in range(max_label):
            mi = tf.multiply(tf.ones_like(p), self.colormap[i])
            m = tf.where(tf.equal(p, i), mi, m)
        return m


class CityscapeDataset(GenericDataset):
    DATASET_NAME = "cityscapes"

    def __init__(self):
        super(CityscapeDataset, self).__init__(dataset_name=CityscapeDataset.DATASET_NAME)
        self.colormap = np.asarray([
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ])
        self.names = [
            'road',
            'sidewalk',
            'building',
            'wall',
            'fence',
            'pole',
            'traffic light',
            'traffic sign',
            'vegetation',
            'terrain',
            'sky',
            'person',
            'rider',
            'car',
            'truck',
            'bus',
            'train',
            'motorcycle',
            'bicycle'
        ]


class PascalDataset(GenericDataset):
    DATASET_NAME = "pascal"

    def __init__(self):
        super(PascalDataset, self).__init__(
            dataset_name=PascalDataset.DATASET_NAME)
