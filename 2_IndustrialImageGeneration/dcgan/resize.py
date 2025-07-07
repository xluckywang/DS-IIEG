import os
from PIL import Image

def resize_images_and_save_to_new_folder(source_directory, target_directory, image_size=(224, 224)):
    # Traverse all files and subfolders in the source directory
    for subdir, _, files in os.walk(source_directory):
        for file in files:
            # Determine whether the file is an image
            _, file_extension = os.path.splitext(file)
            if file_extension.lower() not in ['.jpg', '.jpeg', '.png', '.JPG']:
                continue

            # Obtain the complete path of the image
            source_image_path = os.path.join(subdir, file)

            # Build target folder path (maintain original directory structure)
            relative_path = os.path.relpath(subdir, source_directory)
            target_subdir = os.path.join(target_directory, relative_path)

            # Create target folder
            if not os.path.exists(target_subdir):
                os.makedirs(target_subdir)

            # Obtain the path of the target image
            target_image_path = os.path.join(target_subdir, file)

            # Open the image and resize it
            try:
                with Image.open(source_image_path) as img:
                    img_resized = img.resize(image_size)

                    # Save the resized image to the target path
                    img_resized.save(target_image_path)
                    # print(f"Save the resized image to the target path: {target_image_path}")
            except Exception as e:
                print(f"Unable to process file: {source_image_path}: {e}")

# Example usage
# source_directory = '../data/datasetA1/MTD_5/train/MT_Uneven'  # Replace with source image directory path
source_directory = '../data/datasets_original/NEU_DET/crazing'  # Replace with source image directory path
target_directory = 'datasets/NEU_DET/crazing'  # Replace with the target save directory path

resize_images_and_save_to_new_folder(source_directory, target_directory)