import os
from PIL import Image
from classification import Classification

# Initialize classifier
classification = Classification()

def predict_single_image(image_path):
    """
    Predict the category of a single image
    """
    try:
        image = Image.open(image_path)
        class_name = classification.detect_image(image)
        return class_name
    except Exception as e:
        print(f"Open Error for {image_path}: {e}")
        return None

def predict_images_in_folder(folder_path, output_file):
    """
    预测文件夹中的所有图片并将结果保存到txt文件中
    """
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if os.path.isdir(output_file):
        output_file = os.path.join(output_file, 'prediction_results.txt')

    with open(output_file, 'w') as file:
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            class_name = predict_single_image(image_path)
            if class_name is not None:
                # 将预测结果写入文件
                file.write(f"{image_file}: {class_name}\n")
                print(f"{image_file}: {class_name}")

if __name__ == "__main__":
    # 直接在代码中指定文件夹路径和输出文件路径
    folder_path = '/home/ie/liyan_private/MyData/crack'
    output_file = '/home/ie/liyan_private/classification-pytorch-main/dataPredict/prediction_results.txt'

    predict_images_in_folder(folder_path, output_file)