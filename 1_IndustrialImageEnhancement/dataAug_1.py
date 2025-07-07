# Example of ordinary rotation enhancement Ori to Ori1
import os
import shutil
import numpy as np
from PIL import Image, ImageEnhance, ImageChops
from tqdm import tqdm


# Data augmentation operation
def move(img):
    offset = ImageChops.offset(img, np.random.randint(1, 200), np.random.randint(1, 400))
    return offset

def flip(img):
    if np.random.randint(1, 3) == 1:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        return img.transpose(Image.FLIP_LEFT_RIGHT)

def rotation(img):
    angle = np.random.choice([90, 180, 270])  # Avoid the rotation angle of the white edge
    return img.rotate(angle, expand=False)

def color(img):
    img = ImageEnhance.Color(img).enhance(np.random.uniform(0.5, 1.5))
    img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.8, 1.5))
    img = ImageEnhance.Contrast(img).enhance(np.random.uniform(1.0, 1.3))
    img = ImageEnhance.Sharpness(img).enhance(np.random.uniform(0.5, 3.0))
    return img

def crop(img):
    w, h = img.size
    crop_box = (w / np.random.randint(10, 50), h / np.random.randint(20, 50),
                w * (np.random.randint(9, 49) / 50), h * (np.random.randint(19, 49) / 50))
    cropped = img.crop(crop_box)
    return cropped.resize((w, h))

def random_run(probability, func, img):
    if np.random.rand() < probability / 100.0:
        return func(img)
    return img

# Dataset path and category
DATASET_PATH = "../0_data/datasetOri/MTD/train"
OUTPUT_PATH = "../0_data/datasetAug/MTD/train1"
CLASS_NAMES = ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Uneven','MT_Free'] # Replace with the actual category name
TARGET_COUNTS = [100, 100, 100,100,100,100,100] # Number of targets for each category (including raw data)
os.makedirs(OUTPUT_PATH, exist_ok=True)
for class_name in CLASS_NAMES:
    os.makedirs(os.path.join(OUTPUT_PATH, class_name), exist_ok=True)

for class_name, target_count in zip(CLASS_NAMES, TARGET_COUNTS):
    input_folder = os.path.join(DATASET_PATH, class_name)
    output_folder = os.path.join(OUTPUT_PATH, class_name)
    images = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.JPG','.jpg', '.jpeg'))]
    original_count = len(images)

    # Copy the original image
    for img_name in tqdm(images, desc=f"Copy {class_name} Original image"):
        shutil.copy(os.path.join(input_folder, img_name),
                    os.path.join(output_folder, img_name))

    if original_count >= target_count:
        print(f"{class_name} Target quantity met, no need to expand")
        continue

    generated_count = 0
    with tqdm(total=target_count - original_count, desc=f"augment {class_name}") as pbar:
        while original_count + generated_count < target_count:
            img_name = np.random.choice(images)
            img_path = os.path.join(input_folder, img_name)
            img = Image.open(img_path)

            img = random_run(60, flip, img)
            img = random_run(60, crop, img)
            img = random_run(60, move, img)
            img = random_run(60, rotation, img)
            img = random_run(90, color, img)

            save_name = f"aug_{generated_count}_{img_name}"
            img.save(os.path.join(output_folder, save_name))
            generated_count += 1
            pbar.update(1)

    print(f"Class: {class_name} Augment completed, final quantity: {original_count + generated_count}")