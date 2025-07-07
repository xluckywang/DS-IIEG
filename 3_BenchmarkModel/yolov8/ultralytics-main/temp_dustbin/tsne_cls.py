# 一条线
import os
import torch
from ultralytics import YOLO
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# 加载训练好的 YOLOv8 分类模型
# model = YOLO('yolov8n-cls.pt')  # 替换为你的模型路径
model = YOLO('runs/classify/train/weights/best.pt')  # 替换为你的模型路径
# print(f"Model loaded: {model}")
# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# 加载数据集
def load_images_from_folder(folder):
    images = []
    labels = []
    for label_name in os.listdir(folder):
        label_path = os.path.join(folder, label_name)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                images.append(img)
                labels.append(label_name)
    return torch.stack(images), labels

train_folder = '../data/data_tsne/B/lhj_train_multi'  # 替换为你的训练集路径
images, labels = load_images_from_folder(train_folder)

# 提取特征
model.eval()
features = []
with torch.no_grad():
    for img in images:
        outputs = model(img.unsqueeze(0), verbose=False)  # 禁用 verbose 输出
        prob_vector = outputs[0].probs.data.cpu().numpy()  # 提取概率向量并转换为 NumPy 数组
        features.append(prob_vector)
features = np.array(features)

# 检查特征形状
print(f"Features shape: {features.shape}")

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42, perplexity=30)  # 设置 perplexity 为 30
features_tsne = tsne.fit_transform(features)

# 可视化
plt.figure(figsize=(10, 8))
for i, label in enumerate(set(labels)):
    indices = [j for j, l in enumerate(labels) if l == label]
    plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], label=label)
plt.legend()
plt.title('t-SNE visualization of YOLOv8 classification features')
plt.show()