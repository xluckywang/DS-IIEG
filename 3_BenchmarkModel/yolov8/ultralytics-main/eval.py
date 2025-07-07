from ultralytics import YOLO
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 加载训练好的 YOLOv8 分类模型
# model = YOLO("runs/classify/train_lhj_two_A/weights/best.pt")  # 替换为你的模型权重路径
model = YOLO("runs/classify/train_MTD_50_0418_A/weights/best.pt")  # 替换为你的模型权重路径

# 2. 定义测试集路径
test_path = "../data/datasetA/MTD_50_0418/test"  # 替换为你的测试集根目录

# 3. 从文件夹结构中提取类别名称和真实标签
class_names = sorted(os.listdir(test_path))  # 获取所有类别名称
true_labels = []
test_images = []

for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(test_path, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        test_images.append(img_path)
        true_labels.append(class_idx)  # 类别索引作为真实标签

# 4. 对测试集进行推理
pred_labels = []
for img_path in test_images:
    results = model(img_path)  # 对每张图片进行推理
    pred = results[0].probs.top1  # 获取预测类别的索引
    pred_labels.append(pred)

# 5. 计算全局指标
accuracy = accuracy_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels, average="macro")
precision = precision_score(true_labels, pred_labels, average="macro")
f1 = f1_score(true_labels, pred_labels, average="macro")

print(f"Global Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

# 6. 计算每个类别的 accuracy
class_accuracy = {}
for class_idx, class_name in enumerate(class_names):
    # 获取该类别的样本索引
    class_mask = np.array(true_labels) == class_idx
    # 计算该类别的 accuracy
    class_acc = accuracy_score(np.array(true_labels)[class_mask], np.array(pred_labels)[class_mask])
    class_accuracy[class_name] = class_acc
    print(f"Accuracy for {class_name}: {class_acc:.4f}")

# 7. 计算混淆矩阵
conf_matrix = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:")
print(conf_matrix)

# 8. 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()