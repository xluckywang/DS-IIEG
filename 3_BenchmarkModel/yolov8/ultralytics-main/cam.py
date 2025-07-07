import torch
import cv2
import os
from ultralytics import YOLO
from torchvision.transforms import ToTensor

# ✅ 加载模型
model = YOLO('runs/detect/train_lhj_multi_A/weights/best.pt')

# ✅ 获取目标层
target_layer = model.model.model[-2]  # 倒数第二层，您可以根据需要修改

# ✅ 确保模型在 GPU 上
model.cuda()

# ✅ 测试集路径
test_folder = 'data/lhj_train_multi_A/test/images/'
save_folder = 'runs/gradcam/'
os.makedirs(save_folder, exist_ok=True)

# ✅ 遍历所有测试集图片
for file_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, file_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ 跳过无法加载的图片: {img_path}")
        continue

    # ✅ 转换成 tensor
    img_tensor = ToTensor()(img).unsqueeze(0).cuda()

    # ✅ 设置 requires_grad 为 True
    img_tensor.requires_grad = True

    # ✅ 执行模型推理
    model.model.train()  # 启用训练模式
    output = model.model(img_tensor)

    # ✅ 获取预测类别和概率
    pred_scores = torch.nn.functional.softmax(output[0], dim=0)
    pred_class = torch.argmax(pred_scores)  # 预测类别

    # ✅ 打印预测类别和分数，帮助调试
    print("Prediction Scores: ", pred_scores)
    print("Predicted Class: ", pred_class)

    # ✅ 获取对应的分数（确保它是标量）
    scalar_score = pred_scores[0,pred_class,:,:].sum()

    # ✅ 对预测类别进行反向传播
    scalar_score.backward(retain_graph=True)

    # ✅ 提取梯度和特征图
    gradients = target_layer.register_full_backward_hook(lambda mod, grad_in, grad_out: grad_out[0])
    feature_maps = target_layer.register_forward_hook(lambda mod, input, output: output)

    # ✅ 获取 CAM
    cam = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(cam * feature_maps, dim=1).squeeze().detach().cpu().numpy()

    # ✅ 归一化并生成热力图
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap = cv2.applyColorMap((cam * 255).astype('uint8'), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    # ✅ 保存热力图
    save_path = os.path.join(save_folder, file_name)
    cv2.imwrite(save_path, overlay)
    print(f"✅ 已保存: {save_path}")
