import os
import random
from os import getcwd

from utils.utils import get_classes

# -------------------------------------------------------------------#
#   classes_path    指向model_data下的txt，与自己训练的数据集相关
#                   训练前一定要修改classes_path，使其对应自己的数据集
#                   txt文件中是自己所要去区分的种类
#                   与训练和预测所用的classes_path一致即可
# -------------------------------------------------------------------#
classes_path = 'model_data/cls_classes.txt'
# -------------------------------------------------------#
#   datasets_path   指向数据集所在的路径
# -------------------------------------------------------#
# datasets_path = '../dcgan-pytorch-main/results/predicts/mtd_original/train'  # 直接指向train文件夹
datasets_path = '../data/data_tsne/lhj/A2'  # 直接指向train文件夹
# datasets_path = '../data/datasetA/MTD_train_multi/test'  # 直接指向train文件夹

# 生成的txt文件名（可以通过修改变量值来更改文件名）
output_txt_name = 'cls_tsne.txt'  # 这里可以修改为任意文件名

# 获取类别
classes, _ = get_classes(classes_path)

if __name__ == "__main__":
    # 打开文件，准备写入
    list_file = open(output_txt_name, 'w')

    # 遍历train文件夹下的类别文件夹
    types_name = os.listdir(datasets_path)
    for type_name in types_name:
        if type_name not in classes:
            continue
        cls_id = classes.index(type_name)

        # 遍历类别文件夹下的图片
        photos_path = os.path.join(datasets_path, type_name)
        photos_name = os.listdir(photos_path)
        for photo_name in photos_name:
            _, postfix = os.path.splitext(photo_name)
            if postfix not in ['.jpg', '.png', '.jpeg', '.JPG']:
                continue
            # 写入类别ID和图片路径
            list_file.write(str(cls_id) + ";" + '%s' % (os.path.join(photos_path, photo_name)))
            list_file.write('\n')
    list_file.close()

    # 读取文件内容并随机打乱
    with open(output_txt_name, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)  # 随机打乱

    # 将打乱后的内容写回文件
    with open(output_txt_name, 'w') as f:
        f.writelines(lines)