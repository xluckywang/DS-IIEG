import os
from dcgan import DCGAN

def ensure_directory_exists(path):
    """Ensure that the directory exists, and create if it does not exist"""
    os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    # Define basic path
    base_path = "results/predicts/MTD/Blowhole"

    # Ensure that the basic path exists
    ensure_directory_exists(base_path)

    # Define save path
    save_path_5x5 = os.path.join(base_path, "predict_5x5_results.png")
    save_path_1x1 = os.path.join(base_path, "predict_1x1_results.png")

    dcgan = DCGAN() #Modify the generated path used inside

    num_images = 10  # Number of generated 1x1 images

    # Generate 5x5 image (only need to generate once)
    # dcgan.generate_5x5_image(save_path_5x5)

    # Generate multiple 1x1 images
    for i in range(num_images):
        # Generate 1x1 images with file names sorted by sequence number
        image_path = os.path.join(base_path, f"predict_1x1_results_{i + 1}.png")
        dcgan.generate_1x1_image(image_path)
        print(f"The {i+1} th 1x1 image has been generated, save path: {image_path}")  # Print generation progress and path