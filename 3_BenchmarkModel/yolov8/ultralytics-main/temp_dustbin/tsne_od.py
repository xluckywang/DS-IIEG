import os
import torch
from ultralytics import YOLO
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# 加载训练好的 YOLOv8 目标检测模型
model = YOLO('yolov8n.pt')  # 替换为你的目标检测模型路径
model.eval()

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # YOLOv8 默认输入大小为 640x640
    transforms.ToTensor(),
])

# 加载数据集
def load_images_from_folder(folder):
    """
    加载分类存放的数据集，返回图像张量和对应的类别标签。
    """
    images = []
    labels = []  # 存储每个图像的类别标签
    for label_name in os.listdir(folder):  # 遍历每个类别文件夹
        label_path = os.path.join(folder, label_name)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                images.append(img)
                labels.append(label_name)  # 记录类别标签
    return torch.stack(images), labels

data_folder = '../data/data_tsne/B/lhj_train_multi'  # 替换为你的数据集路径
images, labels = load_images_from_folder(data_folder)

# 提取特征
features = []
# 使用钩子提取中间层特征
features_hook = []

def hook_fn(module, input, output):
    """
    钩子函数，用于提取中间层特征。
    """
    if isinstance(output, tuple):
        features_hook.append(output[0].cpu().numpy())  # 提取张量并转换为 NumPy
    else:
        features_hook.append(output.cpu().numpy())

# 注册钩子到模型的骨干网络（backbone）的最后一层
hook = model.model.model[-1].register_forward_hook(hook_fn)

with torch.no_grad():
    for img in images:
        _ = model(img.unsqueeze(0))  # 推理
        # 提取特征（每次推理后需要从钩子中取出特征）
        if features_hook:
            pass  # 特征已经在钩子中提取并存储

# 假设 features_hook 中存储了特征
features = np.vstack(features_hook)

# 检查特征形状
print(f"Extracted features shape: {features.shape}")

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42, perplexity=30)  # 设置 perplexity 为 30
features_tsne = tsne.fit_transform(features[:100])  # 只取前 100 个样本进行可视化

# 可视化
plt.figure(figsize=(10, 8))
for i in range(features_tsne.shape[0]):
    plt.scatter(features_tsne[i, 0], features_tsne[i, 1], label=labels[i])  # 使用类别标签
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 将图例移到图像外
plt.title('t-SNE visualization of YOLOv8 detection features')
plt.show()

# 移除钩子
hook.remove()