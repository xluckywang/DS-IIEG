import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import get_classes
from nets import get_model_from_name
import os
import argparse

# -----------------------------------------------------------------#
# 1. 提取特征向量
# -----------------------------------------------------------------#
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            features_batch = model(imgs)
            features.append(features_batch.cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


# -----------------------------------------------------------------#
# 2. t-SNE可视化
# -----------------------------------------------------------------#
def plot_tsne(features, labels, class_names, title, save_dir):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(np.unique(labels)):
        indices = labels == label
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=class_names[label], alpha=0.7)

    plt.legend()
    # plt.title(f't-SNE of Features - {title}')

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot as a PNG image
    plot_path = os.path.join(save_dir, f'tsne_{title}.png')
    plt.savefig(plot_path)
    plt.close()  # Close the plot to avoid memory issues


# -----------------------------------------------------------------#
# 3. 主程序
# -----------------------------------------------------------------#
if __name__ == "__main__":
    # ----------------------------#
    # 设置命令行参数
    # ----------------------------#
    parser = argparse.ArgumentParser(description="t-SNE Visualization for Feature Embeddings")
    parser.add_argument('--classes_path', type=str, default='model_data/cls_classes.txt', help='Path to class names file')
    parser.add_argument('--train_annotation_path', type=str, default='cls_tsne.txt', help='Path to training annotation file')
    parser.add_argument('--tsne_save_root', type=str, default='results_tsne/MTD/Vgg/Ori', help='Root directory to save t-SNE plots')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['mobilenet', 'resnet50', 'vgg16', 'vit', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base', 'cnn_transformer'], help='Backbone model name')
    parser.add_argument('--model_path', type=str, default='results_train/Ori/MTD/vgg16/best_epoch_weights.pth', help='Path to pre-trained model weights')
    args = parser.parse_args()

    # ----------------------------#
    # 设置设备、模型和数据集
    # ----------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names, _ = get_classes(args.classes_path)

    # 选择模型
    model = get_model_from_name[args.backbone](num_classes=len(class_names), pretrained=True)
    model.to(device)

    # 加载训练集
    train_lines = open(args.train_annotation_path, 'r').readlines()
    train_dataset = DataGenerator(train_lines, input_shape=[224, 224], random=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=detection_collate)

    # ----------------------------#
    # 加载权重前的特征提取并可视化
    # ----------------------------#
    features_before, labels_before = extract_features(model, train_loader, device)
    plot_tsne(features_before, labels_before, class_names, "Before Loading Weights", args.tsne_save_root)

    # ----------------------------#
    # 加载预训练权重
    # ----------------------------#
    if args.model_path:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location=device)
        model_dict.update(pretrained_dict)  # 更新模型的参数
        model.load_state_dict(model_dict)

    # ----------------------------#
    # 加载权重后的特征提取并可视化
    # ----------------------------#
    features_after, labels_after = extract_features(model, train_loader, device)
    plot_tsne(features_after, labels_after, class_names, "After Loading Weights", args.tsne_save_root)