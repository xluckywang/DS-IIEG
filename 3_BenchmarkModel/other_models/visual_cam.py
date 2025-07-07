# 改好
import os
import random
import torch
import numpy as np
import cv2
from torchvision import transforms
from utils.utils import get_classes
from nets import get_model_from_name


# 固定随机种子，保证复现性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Grad-CAM实现
class GradCAM:
    def __init__(self, model, target_layer, model_name):
        self.model = model
        self.model_name = model_name
        self.target_layer = target_layer
        self.gradient = None
        self.activations = None

        self.model.eval()  # 保证在eval模式下

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def generate(self, input_image, class_idx=None):
        input_image = input_image.unsqueeze(0)
        self.model.zero_grad()
        output = self.model(input_image)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradient[0].cpu().detach().numpy()
        activations = self.activations[0].cpu().detach().numpy()

        if gradients.ndim == 3:  # CNN
            weights = np.mean(gradients, axis=(1, 2))
            cam = np.sum(weights[:, None, None] * activations, axis=0)

        elif gradients.ndim == 2:  # Transformer
            weights = np.mean(gradients, axis=0)
            cam = np.dot(activations, weights)

            if 'vit' in self.model_name.lower():
                cam = cam[1:]  # 去掉CLS token
                cam = cam.reshape(14, 14)
            else:
                side = int(np.sqrt(cam.shape[0]))
                cam = cam.reshape(side, side)

        else:
            raise ValueError(f"Unsupported gradients shape: {gradients.shape}")

        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam=1-cam
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))

        return cam

    def overlay(self, original_image, cam):
        if isinstance(original_image, torch.Tensor):
            original_image = original_image.cpu().numpy().transpose((1, 2, 0))
            original_image = np.uint8(original_image * 255)

        # original_image = cv2.resize(original_image, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        return overlay


def process_images_in_directory(input_dir, output_dir, model, grad_cam, device):
    os.makedirs(output_dir, exist_ok=True)

    preprocess = transforms.Compose([
        # transforms.ToTensor(),
        transforms.ToTensor(),  # 将 PIL Image (HWC, 0-255, 尺寸224x224) 转为 Tensor (CHW, 0-1, 尺寸224x224)
        transforms.Lambda(lambda x: x * 2 - 1)  # 规范化到 [-1, 1]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            print(f"Processing {filename}...")
            img_path = os.path.join(input_dir, filename)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            # img_resized = cv2.resize(img, (224, 224))
            # img_tensor = preprocess(img_resized).to(device)
            img_tensor = preprocess(img).to(device)

            cam = grad_cam.generate(img_tensor)
            # overlay_image = grad_cam.overlay(img_resized, cam)
            overlay_image = grad_cam.overlay(img, cam)

            cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
            print(f"Saved Grad-CAM result for {filename}")


def get_target_layer(model, backbone):
    if backbone == 'vgg16':
        return model.features[-1]
    elif backbone == 'mobilenet':
        return model.features[-1][0]
    elif backbone == 'resnet50':
        return model.layer4[-1].conv3
    elif backbone == 'vit':
        return model.blocks[-1].norm1
    elif backbone in ['swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
        # return model.layers[-1].blocks[-1].norm1
        return model.norm
    elif backbone == 'cnn_transformer':
        # 对于 cnn_transformer，我们选择 CNNFeatureExtractor 部分的最后一个卷积层（index 6）
        return model.cnn_feature_extractor.cnn[6]
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")


if __name__ == "__main__":
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes_path = 'model_data/cls_classes.txt'
    class_names, _ = get_classes(classes_path)

    backbone = 'vgg16'  # 选用backbone
    model = get_model_from_name[backbone](num_classes=len(class_names), pretrained=True)
    model.to(device)
    model.eval()

    model_path = "results_train/Ori/MTD/vgg16/xxx.pth"
    if model_path:
        pretrained_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(pretrained_dict)

    target_layer = get_target_layer(model, backbone)
    grad_cam = GradCAM(model, target_layer, model_name=backbone)

    input_directory = "../../0_data/datasetOri/MTD/train/Blowhole"
    output_directory = "results_cam/MTD/vgg16/Blowhole"

    process_images_in_directory(input_directory, output_directory, model, grad_cam, device)