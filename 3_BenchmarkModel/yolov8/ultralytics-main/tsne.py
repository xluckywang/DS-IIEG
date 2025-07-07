import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
from ultralytics import YOLO
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image

# 忽略警告
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

def preprocess_image(image_path, img_size=224):
    """预处理图像，将其调整为模型输入大小"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size))
    image = np.float32(image) / 255  # 归一化
    return image

def extract_features(model, image_path):
    """从模型中提取特征"""
    image = preprocess_image(image_path)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # 转换为张量并增加batch维度
    with torch.no_grad():
        # 使用 EigenCAM 提取特征
        cam = EigenCAM(model, target_layers=[model.model.model[-2]], task='cls')  # cls, od
        features = cam(image)[0]  # 提取特征
    return features.flatten()  # 展平并转换为numpy数组

# 定义模型权重和对应的数据集路径
model_weights = [
    'yolov8n-cls.pt',
    'runs/classify_MTD_50_0418/train_MTD_50_0418_A/weights/best.pt',
    'runs/classify_MTD_50_0418/train_MTD_50_0418_B/weights/best.pt',
    'runs/classify_MTD_50_0418/train_MTD_50_0418_C/weights/best.pt',
]
train_folders = [
    '../data/datasetA/MTD_50_0418/train',
    '../data/datasetA/MTD_50_0418/train',
    '../data/datasetB/MTD_50_0418/train',
    '../data/datasetC/MTD_50_0418/train',
]
# 输出文件夹
output_folder = 'results_tsne/mtd'
os.makedirs(output_folder, exist_ok=True)
# 加载所有模型
models = [YOLO(weight, verbose=False).cpu() for weight in model_weights]



# 对每个模型和数据集提取特征并生成 t-SNE 可视化
for model_idx, (model, train_folder) in enumerate(zip(models, train_folders)):
    print(f"Processing model {model_idx + 1}: {model_weights[model_idx]} with dataset: {train_folder}")

    # 动态获取类别名称
    class_names = [d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))]
    class_names.sort()  # 按字母顺序排序

    # 获取所有图片的路径和标签
    image_paths = []
    labels = []
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(train_folder, class_name)
        for image_name in os.listdir(class_folder):
            if image_name.endswith(('.png', '.jpg', '.jpeg','JPG')):
                image_paths.append(os.path.join(class_folder, image_name))
                labels.append(class_idx)

    labels = np.array(labels)

    # 提取特征
    features = []
    for image_path in image_paths:
        feature = extract_features(model, image_path)
        features.append(feature)
    features = np.array(features)
    print("=========================================================")
    print(f"数据集样本数{features.shape[0]}")

    # 使用 t-SNE 进行降维
    # tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(features)

    # 可视化 t-SNE 结果
    plt.figure(figsize=(10, 8))
    for class_idx, class_name in enumerate(class_names):
        plt.scatter(tsne_results[labels == class_idx, 0], tsne_results[labels == class_idx, 1], label=class_name)

    # plt.title(f't-SNE Visualization of Features (Model {model_idx + 1})')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    plt.legend(loc='upper right', frameon=False)
    # plt.legend()

    # 保存 t-SNE 图
    output_path = os.path.join(output_folder, f'tsne_model_{model_idx + 1}.png')
    plt.savefig(output_path)
    plt.savefig(output_path,bbox_inches='tight')
    plt.close()

    print(f"t-SNE visualization saved to: {output_path}")

print("Batch processing complete!")