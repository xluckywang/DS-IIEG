# Can generate process diagrams for frequency domain enhancement
import os
import cv2
import numpy as np
import matplotlib # Import specifically for backend setting
matplotlib.use('Agg') # Use a non-interactive backend for matplotlib if GUI is an issue, or ensure one is available
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import argparse
import random

# --- Helper functions (MUST BE TOP-LEVEL for multiprocessing) ---

def frequency_enhance_with_d0(image, D0_val):
    """Frequency domain enhancement (high pass filtering)"""
    if image is None:
        # print(f"Error: Input image to frequency_enhance_with_d0 is None.")
        return None
    if len(image.shape) < 2:
        # print(f"Error: Input image to frequency_enhance_with_d0 is not a valid image array.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    filter_mask = np.ones((rows, cols), np.uint8)
    y1, y2 = max(0, crow - D0_val), min(rows, crow + D0_val)
    x1, x2 = max(0, ccol - D0_val), min(cols, ccol + D0_val)
    filter_mask[y1:y2, x1:x2] = 0

    fft_filtered = fft_shifted * filter_mask
    enhanced_fft_result = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))
    enhanced_normalized = cv2.normalize(enhanced_fft_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return enhanced_normalized

def process_image_with_d0(args_tuple):
    """Processing a single image (multiprocessing worker function)"""
    input_path, output_path_blended, alpha, resize_dim_local, d0_val = args_tuple

    try:
        img_original_from_disk = cv2.imread(input_path)
        if img_original_from_disk is None:
            # print(f"Warning: 无法读取图片: {input_path}. Skipping.")
            return False

        img_to_process = img_original_from_disk.copy() # Start with a copy of the original

        if resize_dim_local:
            if not (isinstance(resize_dim_local, tuple) and len(resize_dim_local) == 2 and resize_dim_local[0] > 0 and resize_dim_local[1] > 0):
                # print(f"Warning: Invalid resize_dim {resize_dim_local} for {input_path}. Using original size.")
                # img_to_process remains the original size
                pass
            else:
                # This is the image that will be used for enhancement AND as the "original" for blending
                img_to_process = cv2.resize(img_original_from_disk, resize_dim_local, interpolation=cv2.INTER_AREA if img_original_from_disk.shape[0]*img_original_from_disk.shape[1] > resize_dim_local[0]*resize_dim_local[1] else cv2.INTER_CUBIC)

        # ---New: Save resized/original image copy ---
        output_dir = os.path.dirname(output_path_blended)
        base_name_orig, ext_orig = os.path.splitext(os.path.basename(input_path))
        output_path_original_copy = os.path.join(output_dir, f"{base_name_orig}_original{ext_orig}")
        cv2.imwrite(output_path_original_copy, img_to_process) # Save the (potentially resized) image
        # --- End adding ---

        min_dim_for_filter = 2
        if img_to_process.shape[0] < max(min_dim_for_filter, d0_val) or \
           img_to_process.shape[1] < max(min_dim_for_filter, d0_val):
            # print(f"Warning: Image {os.path.basename(input_path)} (dims: {img_to_process.shape[:2]}) too small for D0={d0_val}. Skipping enhancement.")
            enhanced_gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
        else:
            enhanced_gray = frequency_enhance_with_d0(img_to_process, D0_val=d0_val)
            if enhanced_gray is None:
                # print(f"Warning: Frequency enhancement failed for {os.path.basename(input_path)}. Using original gray.")
                enhanced_gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)

        enhanced_colored = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        output_path_freq_enhanced = os.path.join(output_dir, f"{base_name_orig}_freq_enhanced{ext_orig}")
        cv2.imwrite(output_path_freq_enhanced, enhanced_colored)

        # Use img_to_process (which is the resized original) for blending
        combined = cv2.addWeighted(img_to_process, 1-alpha, enhanced_colored, alpha, 0)
        cv2.imwrite(output_path_blended, combined) # This is the blended image, saved without suffix
        return True
    except Exception as e:
        # print(f"dealing with failures: {os.path.basename(input_path)}: {str(e)}")
        return False

# cli_args = None # Already defined in the provided code snippet

def visualize_samples(input_root, output_root, alpha, num_samples=3, d0_for_title=None):
    """Randomly display comparison samples before and after processing (original image, pure frequency domain enhanced image, fused image)"""
    if num_samples <= 0:
        return
    try:
        pass # plt is already imported
    except ImportError:
        print("Matplotlib not found. Skipping visualization.")
        return

    class_dirs = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]
    if not class_dirs:
        print("Visualization: No category directory was found in the output directory!")
        return

    valid_img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    save_visualization_to_file = not plt.get_backend() or plt.get_backend().lower() == 'agg'

    # We will display:
    # 1. Original (from input_root, or the _original copy from output_root)
    # 2. Frequency Enhanced (from output_root)
    # 3. Blended (from output_root)
    # Let's stick to showing the *absolute* original from input_root for the first panel,
    # and the two processed images from output_root for the other panels.
    # This makes the "Original" truly original, before any resizing.
    # If you want to show the _original copy from output_root, change orig_path logic.

    fig, axes = plt.subplots(num_samples, 3, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = np.array([axes])

    plotted_samples_count = 0
    attempts = 0
    max_attempts = num_samples * len(class_dirs) * 2 if class_dirs else num_samples * 3

    sample_candidates = []
    for class_name_vis in class_dirs:
        output_class_dir_vis = os.path.join(output_root, class_name_vis)
        # Look for blended images (without suffix) to base the sample selection on
        blended_img_list_vis = [f for f in os.listdir(output_class_dir_vis)
                                if f.lower().endswith(valid_img_extensions) and '_freq_enhanced' not in f and '_original' not in f]
        for blended_name in blended_img_list_vis:
            sample_candidates.append((class_name_vis, blended_name))

    if not sample_candidates:
        print("Visualization: No processed (fused) images available for sampling were found in the output directory.")
        plt.close(fig)
        return

    random.shuffle(sample_candidates)

    for i in range(min(num_samples, len(sample_candidates))):
        class_name, blended_img_name = sample_candidates[i]

        # Path for the *absolute* original image from the input directory
        # This one is NOT resized by our script for display here.
        orig_path_from_input = os.path.join(input_root, class_name, blended_img_name)

        # Paths for images in the output directory
        output_class_dir = os.path.join(output_root, class_name)
        base_name, ext = os.path.splitext(blended_img_name)

        # Path to the saved _original copy (potentially resized)
        # original_copy_path_from_output = os.path.join(output_class_dir, f"{base_name}_original{ext}")

        freq_enhanced_path = os.path.join(output_class_dir, f"{base_name}_freq_enhanced{ext}")
        blended_path = os.path.join(output_class_dir, blended_img_name) # Blended has no suffix

        # For display, we'll use orig_path_from_input as the "Original"
        # If you prefer to show the _original copy (which might be resized), use original_copy_path_from_output

        paths_to_check = [orig_path_from_input, freq_enhanced_path, blended_path]

        if not all(os.path.exists(p) for p in paths_to_check):
            # print(f"Skipping sample {class_name}/{blended_img_name} due to missing files.")
            continue

        try:
            # Load the absolute original from input_root
            orig_img_bgr_abs = cv2.imread(orig_path_from_input)
            # Load processed images from output_root
            freq_enhanced_img_bgr = cv2.imread(freq_enhanced_path)
            blended_img_bgr = cv2.imread(blended_path)

            if any(img is None for img in [orig_img_bgr_abs, freq_enhanced_img_bgr, blended_img_bgr]):
                # print(f"Skipping sample {class_name}/{blended_img_name} due to image loading error.")
                continue

            orig_display = cv2.cvtColor(orig_img_bgr_abs, cv2.COLOR_BGR2RGB)
            freq_enhanced_display = cv2.cvtColor(freq_enhanced_img_bgr, cv2.COLOR_BGR2RGB)
            blended_display = cv2.cvtColor(blended_img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # print(f"Error loading images for sample {class_name}/{blended_img_name}: {e}")
            continue

        ax_orig, ax_freq, ax_blend = axes[plotted_samples_count]

        ax_orig.imshow(orig_display)
        ax_orig.set_title(f"Original (from input)\n{class_name}/{blended_img_name}", fontsize=8)
        ax_orig.axis('off')

        d0_title_str = f"(D0={d0_for_title})" if d0_for_title is not None else ""
        ax_freq.imshow(freq_enhanced_display)
        ax_freq.set_title(f"Freq. Enhanced {d0_title_str}\n{class_name}/{base_name}_freq_enhanced{ext}", fontsize=8)
        ax_freq.axis('off')

        ax_blend.imshow(blended_display)
        ax_blend.set_title(f"Blended (α={alpha})\n{class_name}/{blended_img_name}", fontsize=8)
        ax_blend.axis('off')

        plotted_samples_count += 1
        if plotted_samples_count >= num_samples:
            break


    if plotted_samples_count == 0:
        print("Failed to load any visualization samples successfully. Please check the input/output directory and files.")
        plt.close(fig)
        return

    for i in range(plotted_samples_count, num_samples): # Hide unused subplots
        for j in range(3):
            fig.delaxes(axes[i][j])

    plt.tight_layout()
    if save_visualization_to_file:
        output_viz_path = os.path.join(output_root, "visualization_samples.png")
        plt.savefig(output_viz_path)
        print(f"The visualization sample has been saved to:{output_viz_path}")
    else:
        plt.show()
    plt.close(fig)


def batch_process_dataset_with_d0(input_root, output_root, alpha, workers, resize_param, samples, d0_param):
    """Batch processing the entire dataset"""
    os.makedirs(output_root, exist_ok=True)
    class_dirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]

    tasks_local = []
    valid_img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    for class_name in class_dirs:
        class_input_dir = os.path.join(input_root, class_name)
        class_output_dir = os.path.join(output_root, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        for img_name in os.listdir(class_input_dir):
            if img_name.lower().endswith(valid_img_extensions):
                input_path = os.path.join(class_input_dir, img_name)
                # output_path_blended is the path for the final blended image (no suffix)
                output_path_blended = os.path.join(class_output_dir, img_name)
                tasks_local.append((input_path, output_path_blended, alpha, resize_param, d0_param))

    if not tasks_local:
        print("No valid images were found for processing.")
        return

    print(f"start processing {len(tasks_local)} images (workers={workers}, D0={d0_param}, α={alpha}, resize={resize_param})...")

    results = []
    if workers > 0 :
        with multiprocessing.Pool(processes=workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_image_with_d0, tasks_local), total=len(tasks_local), desc="Processing images"))
    else:
        print("Run in single process mode: (workers=0)...")
        for task_item in tqdm(tasks_local, total=len(tasks_local), desc="Processing images (single thread)"):
            results.append(process_image_with_d0(task_item))

    success_count = sum(r for r in results if r is True) # Filter out None or False
    failure_count = len(tasks_local) - success_count # Assuming tasks_local is the total attempted
    print(f"Processing completed! success: {success_count}/{len(tasks_local)}. Failed: {failure_count}.")


    if success_count > 0 and samples > 0:
        print(f"Generating {samples} visual sample ...")
        visualize_samples(input_root, output_root, alpha, samples, d0_for_title=d0_param)
    elif samples == 0:
        print("Visualization skipped (samples=0).")

if __name__ == "__main__":
    current_start_method = multiprocessing.get_start_method(allow_none=True)
    if os.name != 'posix' or current_start_method != 'fork':
        try:
            multiprocessing.set_start_method("spawn", force=False)
        except RuntimeError:
            pass

    parser = argparse.ArgumentParser(description="Dataset frequency domain enhancement processing tool")
    parser.add_argument("--input", type=str, default="../0_data/datasetOri/MTD/train", help="Input dataset path")
    parser.add_argument("--output", type=str, default="../0_data/datasetS1/MTD/train_3_process", help="Output result path") # Changed output path
    parser.add_argument("--alpha", type=float, default=0.3, help="Fusion weight (0-1), default 0.5") # Changed alpha
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2 if os.cpu_count() else 1), help=f"Number of parallel processes (0 for single main thread), default: automatic ({max(1, os.cpu_count() // 2 if os.cpu_count() else 1)})")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=[224, 224], help="Adjust the image size (width and height), for example 224 224 If set to 0, the size will not be adjusted.")
    parser.add_argument("--samples", type=int, default=3, help="Random sampling inspection quantity, default 3. If set to 0, it will not be displayed.")
    parser.add_argument("--d0", type=int, default=5, help="The cutoff frequency of the high pass filter is D0 (half the length of the square area), with a default of 5.") # Changed d0

    cli_args = parser.parse_args()

    actual_resize = None
    if cli_args.resize and not (cli_args.resize[0] == 0 and cli_args.resize[1] == 0):
        if cli_args.resize[0] > 0 and cli_args.resize[1] > 0:
            actual_resize = tuple(cli_args.resize)
        else:
            print(f"Warning: Invalid resize parameter: {cli_args.resize} Will not adjust the size")

    if cli_args.workers < 0:
        print("Warning: The number of workers cannot be negative, one worker will be used.")
        cli_args.workers = 1
    # elif cli_args.workers > os.cpu_count():
        # print(f"警告: 请求的workers数量 ({cli_args.workers}) 大于CPU核心数 ({os.cpu_count()}). 可能导致性能下降。")
        # pass

    print("Parameter configuration:")
    print(f"- Input directory: {cli_args.input}")
    print(f"- Output directory: {cli_args.output}")
    print(f"- Fusion weight α: {cli_args.alpha}")
    print(f"- High pass filter D0: {cli_args.d0}")
    print(f"- Number of processes: {cli_args.workers if cli_args.workers > 0 else '0 (Main Line Single Process)'}")
    print(f"- resize: {actual_resize if actual_resize else 'Maintain the original size'}")
    print(f"- Sampling quantity: {cli_args.samples}")


    os.makedirs(cli_args.output, exist_ok=True)

    batch_process_dataset_with_d0(
        input_root=cli_args.input,
        output_root=cli_args.output,
        alpha=cli_args.alpha,
        workers=cli_args.workers,
        resize_param=actual_resize,
        samples=cli_args.samples,
        d0_param=cli_args.d0
    )