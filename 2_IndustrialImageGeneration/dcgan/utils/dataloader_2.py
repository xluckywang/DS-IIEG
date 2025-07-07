# 灰度处理
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor


class DCganDataset(Dataset):
    def __init__(self, train_lines, input_shape):
        super(DCganDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.input_shape = input_shape

    def __len__(self):
        return self.train_batches

    def preprocess_input(self, image, mean, std):
        image = (image / 255 - mean) / std
        return image

    def __getitem__(self, index):
        index = index % self.train_batches
        # 打开图像为灰度图（1通道）
        image = Image.open(self.train_lines[index].split()[0]).convert('L')
        # resize 到 224x224
        image = image.resize([self.input_shape[1], self.input_shape[0]], Image.BICUBIC)

        # 转为 numpy 数组 (H, W)
        image = np.array(image, dtype=np.float32)
        # 增加通道维度 -> (1, H, W)
        image = np.expand_dims(image, axis=0)

        # 标准化
        image = self.preprocess_input(image, 0.5, 0.5)  # mean=0.5, std=0.5
        return image


def DCgan_dataset_collate(batch):
    images = []
    for image in batch:
        images.append(image)
    images = np.array(images)
    return images