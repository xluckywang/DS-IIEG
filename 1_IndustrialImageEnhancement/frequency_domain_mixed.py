# Only generate frequency domain enhanced fused images
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import argparse
import random

def frequency_enhance(image, D0=5):
    """Frequency domain enhancement (high pass filtering)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-D0:crow+D0, ccol-D0:ccol+D0] = 0  # HPF

    fft_filtered = fft_shifted * mask
    enhanced = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return enhanced

def process_image(args):
    """Processing a single image (multiprocessing worker function)"""
    input_path, output_path, alpha, resize_dim = args

    try:
        # Read and preprocess
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Unable to read image: {input_path}")

        if resize_dim:
            img = cv2.resize(img, resize_dim)

        # Frequency domain enhancement
        enhanced = frequency_enhance(img)
        enhanced_colored = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # weighted fusion
        combined = cv2.addWeighted(img, 1-alpha, enhanced_colored, alpha, 0)

        # Save the Results
        cv2.imwrite(output_path, combined)
        return True
    except Exception as e:
        print(f"dealing with failures: {input_path}: {str(e)}")
        return False

def visualize_samples(input_root, output_root, alpha, num_samples=3):
    """Randomly display comparison samples before and after processing"""
    class_dirs = [d for d in os.listdir(output_root)
                  if os.path.isdir(os.path.join(output_root, d))]

    if not class_dirs:
        print("No category directory found！")
        return

    plt.figure(figsize=(15, 5*num_samples))

    for i in range(num_samples):
        class_name = random.choice(class_dirs)
        img_list = [f for f in os.listdir(os.path.join(output_root, class_name))
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not img_list:
            continue

        img_name = random.choice(img_list)

        orig_path = os.path.join(input_root, class_name, img_name)
        enhanced_path = os.path.join(output_root, class_name, img_name)

        orig = cv2.cvtColor(cv2.imread(orig_path), cv2.COLOR_BGR2RGB)
        enhanced = cv2.cvtColor(cv2.imread(enhanced_path), cv2.COLOR_BGR2RGB)

        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(orig)
        plt.title(f"Original\n{class_name}/{img_name}")
        plt.axis('off')

        plt.subplot(num_samples, 2, 2*i+2)
        plt.imshow(enhanced)
        plt.title(f"Enhanced (α={alpha})\n{class_name}/{img_name}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def batch_process_dataset(input_root, output_root, alpha=0.3, workers=4, resize=None, samples=3):
    """Batch processing the entire dataset"""
    # Create output directory structure
    os.makedirs(output_root, exist_ok=True)
    class_dirs = [d for d in os.listdir(input_root)
                  if os.path.isdir(os.path.join(input_root, d))]

    # Prepare task list
    tasks = []
    for class_name in class_dirs:
        class_input_dir = os.path.join(input_root, class_name)
        class_output_dir = os.path.join(output_root, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        for img_name in os.listdir(class_input_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_path = os.path.join(class_input_dir, img_name)
                output_path = os.path.join(class_output_dir, img_name)
                tasks.append((input_path, output_path, alpha, resize))

    # Multi process processing
    print(f"Start processing: {len(tasks)} images (workers={workers})...")
    with multiprocessing.Pool(workers) as pool:
        results = list(tqdm(pool.imap(process_image, tasks), total=len(tasks)))

    success_count = sum(results)
    print(f"Processing completed! success: {success_count}/{len(tasks)}")

    # Random sampling visualization
    if success_count > 0:
        visualize_samples(input_root, output_root, alpha, samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset frequency domain enhancement processing tool")
    parser.add_argument("--input", type=str, default="../0_data/datasetOri/MTD/train", help="Input dataset path to train")
    parser.add_argument("--output", type=str, default="../0_data/datasetS1/MTD/train_3", help="Output result path")
    parser.add_argument("--alpha", type=float, default=0.3, help="Fusion weight (0-1), default 0.3")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=[224, 224], help="Adjust image size")
    parser.add_argument("--samples", type=int, default=3, help="Random sampling inspection quantity, default 3")

    args = parser.parse_args()

    print("Parameter configuration:")
    print(f"- Input directory: {args.input}")
    print(f"- Output Directory: {args.output}")
    print(f"- Fusion weight: {args.alpha}")
    print(f"- Number of processes: {args.workers}")
    print(f"- resize: {args.resize if args.resize else 'Maintain the original size'}")
    print(f"- Sample quantity: {args.samples}")

    batch_process_dataset(
        input_root=args.input,
        output_root=args.output,
        alpha=args.alpha,
        workers=args.workers,
        resize=tuple(args.resize) if args.resize else None,
        samples=args.samples
    )