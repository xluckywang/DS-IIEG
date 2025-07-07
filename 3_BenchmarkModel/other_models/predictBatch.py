import os
from PIL import Image
from classification import Classification
import shutil

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

def predict_images_in_folder(folder_path, output_file, crack_folder, ok_folder):
    """
    Predict all the images in the folder and save the results to a txt file, and place the predicted images with a crack in one folder and OK in another folder
    """
    # Retrieve all image files from the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if os.path.isdir(output_file):
        output_file = os.path.join(output_file, 'prediction_results.txt')

    with open(output_file, 'w') as file:
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            class_name = predict_single_image(image_path)
            if class_name is not None:
                # Write the predicted results to a file
                file.write(f"{image_file}: {class_name}\n")
                print(f"{image_file}: {class_name}")

                # Move files based on predicted results
                if class_name == 'crack':
                    shutil.move(image_path, os.path.join(crack_folder, image_file))
                elif class_name == 'ok':
                    shutil.move(image_path, os.path.join(ok_folder, image_file))

if __name__ == "__main__":
    # Specify the folder path and output file path directly in the code
    folder_path = '/home/ie/liyan_private/classification-pytorch-main/img/genCrack'
    output_file = '/home/ie/liyan_private/classification-pytorch-main/img/genOut/prediction_results.txt'
    crack_folder = os.path.join(folder_path, "crack")
    ok_folder = os.path.join(folder_path, "ok")

    # Create target folder
    os.makedirs(crack_folder, exist_ok=True)
    os.makedirs(ok_folder, exist_ok=True)

    predict_images_in_folder(folder_path, output_file, crack_folder, ok_folder)