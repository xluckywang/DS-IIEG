# Example of ordinary rotation enhancement Ori to Ori1
import os
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Augmentation functions
def random_rotate_90(img):
    angle = random.choice([90, 180, 270])
    return img.rotate(angle, expand=True)

def random_flip(img):
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    if random.random() > 0.5:
        img = ImageOps.flip(img)
    return img

def random_color_jitter(img):
    enhancers = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color, ImageEnhance.Sharpness]
    for E in enhancers:
        factor = random.uniform(0.8, 1.2)
        img = E(img).enhance(factor)
    return img

def slight_blur(img):
    if random.random() > 0.5:
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    return img

# Combine augmentations
def augment(img):
    ops = [random_rotate_90, random_flip, random_color_jitter, slight_blur]
    random.shuffle(ops)
    for op in ops[:random.randint(2, 4)]:
        img = op(img)
    return img

def augment_class(source_class_dir, target_class_dir, target_count):
    ensure_dir(target_class_dir)
    image_paths = [os.path.join(source_class_dir, f) for f in os.listdir(source_class_dir)
                   if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    count = 0

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        basename = os.path.basename(path)
        img.save(os.path.join(target_class_dir, basename))
        count += 1

    while count < target_count:
        src_path = random.choice(image_paths)
        img = Image.open(src_path).convert("RGB")
        aug = augment(img)
        name, ext = os.path.splitext(os.path.basename(src_path))
        new_name = f"{name}_aug_{count}{ext}"
        aug.save(os.path.join(target_class_dir, new_name))
        count += 1

    print(f"Class '{os.path.basename(target_class_dir)}' expanded to {count} images.")

def main(dataset_path, target_path, class_counts):
    train_path = os.path.join(dataset_path, "train")
    target_train_path = os.path.join(target_path, "train")
    ensure_dir(target_train_path)

    for class_name, target_count in class_counts.items():
        source_class_dir = os.path.join(train_path, class_name)
        target_class_dir = os.path.join(target_train_path, class_name)
        augment_class(source_class_dir, target_class_dir, target_count)

    print(f"Dataset expanded to the target counts in '{target_path}'.")

if __name__ == "__main__":
    # 修改以下路径和类别目标数量
    dataset_path ="../0_data/datasetOri/MTD"  # 数据集根路径
    result_path = "../0_data/datasetAug/MTD"    # 结果保存路径
    class_counts = {
        'MT_Blowhole'   :100, # 类别1扩充到100张
        'MT_Break'      :100,
        'MT_Crack'      :100,
        'MT_Fray'       :100,
        'MT_Uneven'     :100,
        'MT_Free'       :100
    }
    main(dataset_path, result_path, class_counts)