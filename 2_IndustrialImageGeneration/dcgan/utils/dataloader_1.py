import numpy as np
from PIL import Image
import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor

from torchvision import transforms

class DCganDataset(Dataset):
    def __init__(self, train_lines, input_shape):
        super(DCganDataset, self).__init__()

        self.train_lines    = train_lines
        self.train_batches  = len(train_lines)
        self.input_shape    = input_shape

        self.transform = transforms.Compose([
            transforms.Resize((self.input_shape[0], self.input_shape[1])),
            transforms.ToTensor(),  # 将图像转换为Tensor并归一化到[0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
        ])

    def __len__(self):
        return self.train_batches

    def __getitem__(self, index):
        index   = index % self.train_batches
        image   = Image.open(self.train_lines[index].split()[0])

        # 使用transform进行预处理
        image   = self.transform(image)

        return image
# class DCganDataset(Dataset):
#     def __init__(self, train_lines, input_shape):
#         super(DCganDataset, self).__init__()
#
#         self.train_lines    = train_lines
#         self.train_batches  = len(train_lines)
#         self.input_shape    = input_shape
#
#     def __len__(self):
#         return self.train_batches
#
#     def preprocess_input(self, image, mean, std):
#         image = (image/255 - mean)/std
#         return image
#
#     def __getitem__(self, index):
#         index   = index % self.train_batches
#         image   = Image.open(self.train_lines[index].split()[0])
#         image   = cvtColor(image).resize([self.input_shape[1], self.input_shape[0]], Image.BICUBIC)
#
#         image   = np.array(image, dtype=np.float32)
#         image   = np.transpose(self.preprocess_input(image, 0.5, 0.5), (2, 0, 1))
#         return image

def DCgan_dataset_collate(batch):
    images = []
    for image in batch:
        images.append(image)
    images = np.array(images)
    return images
# import numpy as np
# import cv2
# from torch.utils.data.dataset import Dataset
# from utils.utils import cvtColor  # 如果你有自定义颜色转换函数
#
# class DCganDataset(Dataset):
#     def __init__(self, train_lines, input_shape):
#         """
#         初始化数据集
#         :param train_lines: 包含每张图像路径的列表
#         :param input_shape: 图像输入的目标大小，格式为 (height, width, channels)
#         """
#         super(DCganDataset, self).__init__()
#
#         self.train_lines    = train_lines  # 图像路径列表
#         self.train_batches  = len(train_lines)  # 图像数量
#         self.input_shape    = input_shape  # 目标输入大小
#
#     def __len__(self):
#         """
#         返回数据集的大小
#         """
#         return self.train_batches
#
#     def preprocess_input(self, image, mean, std):
#         """
#         图像预处理，将其标准化
#         :param image: 图像
#         :param mean: 图像标准化均值
#         :param std: 图像标准化标准差
#         :return: 处理后的图像
#         """
#         image = (image / 255 - mean) / std
#         return image
#
#     def __getitem__(self, index):
#         """
#         获取指定索引的图像
#         :param index: 图像的索引
#         :return: 处理后的图像数据
#         """
#         index = index % self.train_batches  # 处理批次索引，避免越界
#         image_path = self.train_lines[index].split()[0]  # 获取图像路径
#
#         # 使用 OpenCV 加载图像
#         image = cv2.imread(image_path)
#         if image is None:
#             raise ValueError(f"Failed to load image: {image_path}")
#
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB
#         image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))  # 调整图像大小
#
#         # 转换为 numpy 数组，并进行标准化处理
#         image = np.array(image, dtype=np.float32)
#         image = np.transpose(self.preprocess_input(image, 0.5, 0.5), (2, 0, 1))  # 转换维度 (H, W, C) -> (C, H, W)
#
#         return image
#
# def DCgan_dataset_collate(batch):
#     """
#     数据集 collate 函数，将批次中的数据组合成一个 batch
#     :param batch: 数据列表
#     :return: 合并后的 batch
#     """
#     images = []
#     for image in batch:
#         images.append(image)
#     images = np.array(images)  # 将列表转换为 numpy 数组
#     return images