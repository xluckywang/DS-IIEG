import os
import shutil

# 源文件夹路径
source_folder = "../data/datasetA/MTD_train_two/train"
# 目标文件夹路径
target_folder = "data/MTD_train_two_A_test/train/images"
# 标签文件夹路径
labels_folder = "data/MTD_train_two_A_test/train/labels"
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
if not os.path.exists(labels_folder):
    os.makedirs(labels_folder)
# 缺陷类型列表
# defect_types = ['crack', 'irregular', 'prorosity', 'sagging', 'scratch']
defect_types = ['defective', 'ok']
# 遍历每个缺陷类型文件夹
for i, defect_type in enumerate(defect_types):
    defect_folder = os.path.join(source_folder, defect_type)

    # 遍历缺陷类型文件夹中的图像文件
    for image_file in os.listdir(defect_folder):
        if image_file.endswith(".JPG") or image_file.endswith(".PNG") or image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(defect_folder, image_file)
            target_image_path = os.path.join(target_folder, image_file)
            shutil.copy(image_path, target_image_path)

            # 创建对应的txt文件
            label_content = f"{i} 0.5 0.5 1 1"
            label_file_name = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(labels_folder, label_file_name)

            with open(label_path, 'w') as label_file:
                label_file.write(label_content)