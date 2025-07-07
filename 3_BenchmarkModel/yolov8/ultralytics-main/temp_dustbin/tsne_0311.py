# 不同权重一个结果 不行
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ultralytics import YOLO
from PIL import Image
import os
import torchvision.transforms as transforms

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 直接调用 NumPy 的 `threadpoolctl` 限制线程
try:
    import threadpoolctl
    threadpoolctl.threadpool_limits(limits=1)
except ImportError:
    print("threadpoolctl not installed, skipping.")
#
# # 确保 NumPy 确实生效
# print(f"NumPy OpenBLAS 线程数: {np.__config__.show()}")
# ✅ 1. 加载训练好的 YOLOv8 分类模型
model = YOLO('runs/detect/train_lhj_multi_A/weights/best.pt')
# model = YOLO('runs/detect/train_lhj_multi_A/weights/best.pt')  # 或 'runs/classify/train/weights/best.pt'
# model.eval()
model = model.cpu()
print("Model training mode:", model.training)

# ✅ 2. 预处理（与训练时保持一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小
    transforms.ToTensor(),  # 转换为 Tensor
])

# ✅ 3. 获取数据集路径

data_root = "../data/data_tsne/B/lhj_train_multi"  # 可更改为 train/val
class_names = sorted(os.listdir(data_root))  # 获取类别名称
features = []
labels = []

# ✅ 4. 遍历测试集，提取特征
for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(data_root, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).cuda()  # 预处理并添加 batch 维度

        # ✅ 5. 获取中间层特征
        with torch.no_grad():
            # feature_extractor = torch.nn.Sequential(*list(model.model.children())[:-1])  # 去掉 `Classify` 层
            feature_extractor = torch.nn.Sequential(*list(model.model.children())[:-2])  # 去掉 `Classify` 层
            feature = feature_extractor(img_tensor)

            feature = feature.view(feature.shape[0], -1).cpu().numpy()

        features.append(feature)
        labels.append(class_idx)

# ✅ 6. 转换为 numpy 数组
features = np.array([f.squeeze(0) for f in features])  # 确保 shape = (num_samples, feature_dim)
labels = np.array(labels)

# ✅ 7. t-SNE 降维
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
features_2d = tsne.fit_transform(features)  # 现在 features 形状正确

# ✅ 8. 绘制 t-SNE 结果
# plt.figure(figsize=(10, 8),dpi=600)
plt.figure(figsize=(10, 8))
# scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels,alpha=0.7)
plt.legend(handles=scatter.legend_elements()[0], labels=class_names, title="Classes")
plt.title("t-SNE Visualization of YOLOv8 Classification Features")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
output_dir='results_tsne/lhj_2'
os.makedirs(output_dir, exist_ok=True)
output_path=os.path.join(output_dir, 'A_tsne_od.png')
plt.savefig(output_path)
print(f"Saved t-SNE visualization to {output_path}")