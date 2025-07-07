import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image
from PIL import Image

# 忽略警告
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# 创建保存结果的文件夹
output_folder = 'results_cam/yolo_0609/mtd'
os.makedirs(output_folder, exist_ok=True)

# 定义模型权重路径
model_weights = [
    'yolov8n-cls.pt',
    'runs/classify_MTD_50_0418/train_MTD_50_0418_A/weights/best.pt',
    'runs/classify_MTD_50_0418/train_MTD_50_0418_B/weights/best.pt',
    'runs/classify_MTD_50_0418/train_MTD_50_0418_C/weights/best.pt'
]

# 图片文件夹路径 - 现在指向包含类别子目录的父目录
input_folder = '../data/datasetA/MTD_50_0418/train'


def process_model(model_weight, input_folder, output_folder):
    # 读取训练好的YOLOv8模型
    model = YOLO(model_weight)
    model = model.cpu()
    target_layers = [model.model.model[-2]]
    # 初始化EigenCAM
    cam = EigenCAM(model, target_layers, task='cls')  # cls od seg

    # 遍历每个类别子目录
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        # 在输出目录中创建对应的类别子目录
        class_output_dir = os.path.join(output_folder, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # 获取该类别下的所有图片
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg','JPG'))]

        for image_file in image_files:
            # 读取并处理图片
            img_path = os.path.join(class_path, image_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (640, 640))  # 调整大小
            rgb_img = img.copy()
            img = np.float32(img) / 255  # 归一化

            # 生成Grad-CAM
            grayscale_cam = cam(rgb_img)[0, :, :]
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

            # 保存结果
            model_name = os.path.basename(os.path.dirname(os.path.dirname(model_weight)))
            result_path = os.path.join(class_output_dir, f'{os.path.splitext(image_file)[0]}_{model_name}.png')
            plt.imshow(cam_image)
            plt.axis('off')
            plt.savefig(result_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            print(f"Processed and saved: {result_path}")


# 循环处理每个模型
for model_weight in model_weights:
    process_model(model_weight, input_folder, output_folder)

print("Batch processing complete!")