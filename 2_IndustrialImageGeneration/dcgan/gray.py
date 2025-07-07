import os
import cv2
from tqdm import tqdm  #Progress bar support (optional)

def convert_to_grayscale(src_dir, dst_dir):
    """
    Convert all images in the source directory to grayscale and save them to the target directory
    : paramsrc_dir: Source image directory path (e.g. 'path/to/pictures')
    : paramdst-dir: Target directory path (e.g. 'results/pictures')
    """
    # Ensure that the target directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Supported image formats
    valid_extensions = ('.png', '.jpg', '.jpeg')

    # Get all image files
    image_files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith(valid_extensions)
    ]

    # Process each image (with progress bar)
    for filename in tqdm(image_files, desc="Processing Images"):
        try:
            # Read images (including color and grayscale)
            img_path = os.path.join(src_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            # Convert to grayscale image (if not already grayscale image)
            if len(img.shape) == 3:  # Color Map（H,W,C）
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:  # It is already a grayscale image (H, W)
                gray_img = img

            # Save grayscale image (keep original file name)
            output_path = os.path.join(dst_dir, filename)
            cv2.imwrite(output_path, gray_img)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # input parameter
    source_directory = "results/predicts/XSDD/test/oxide_scale_of_plate_system_original"  # Replace with your source directory
    target_directory = "results/predicts/XSDD/test/oxide_scale_of_plate_system"  # Replace with target directory

    # Perform Transformation
    convert_to_grayscale(source_directory, target_directory)
    print(f"\nThe grayscale image has been saved to: {target_directory}")