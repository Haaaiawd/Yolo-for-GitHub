import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import shutil
import sys
from pathlib import Path

# Add project source to sys.path to allow for ultralytics import
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # This should be the project root 'Yolo'
SOURCE_ROOT = ROOT / 'source' / 'yolo13'
if str(SOURCE_ROOT) not in sys.path:
    sys.path.append(str(SOURCE_ROOT))


def process_sequence(input_image_dir, input_label_dir, output_image_dir, output_label_dir):
    """
    Processes a single sequence to create frame-differenced images and copy corresponding labels.

    Args:
        input_image_dir (str): Path to the directory containing the source image sequence.
        input_label_dir (str): Path to the directory containing the source label sequence.
        output_image_dir (str): Path to the directory where processed images will be saved.
        output_label_dir (str): Path to the directory where processed labels will be saved.
    """
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    image_files = sorted([f for f in os.listdir(input_image_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])

    if len(image_files) < 2:
        print(f"Warning: Not enough images to process in {input_image_dir}. Skipping.")
        return

    for i in tqdm(range(1, len(image_files)), desc=f"Processing {os.path.basename(input_image_dir)}"):
        prev_image_path = os.path.join(input_image_dir, image_files[i-1])
        curr_image_path = os.path.join(input_image_dir, image_files[i])

        prev_img = cv2.imread(prev_image_path, cv2.IMREAD_GRAYSCALE)
        curr_img = cv2.imread(curr_image_path, cv2.IMREAD_GRAYSCALE)

        if prev_img is None or curr_img is None:
            print(f"Warning: Could not read one of the images: {prev_image_path} or {curr_image_path}. Skipping.")
            continue

        diff_img = cv2.absdiff(curr_img, prev_img)
        _, thresh_img = cv2.threshold(diff_img, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        morphed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

        output_filename = image_files[i]
        output_image_path = os.path.join(output_image_dir, output_filename)
        cv2.imwrite(output_image_path, morphed_img)

        # Handle the corresponding label file
        base_filename, _ = os.path.splitext(output_filename)
        label_filename = f"{base_filename}.txt"
        
        source_label_path = os.path.join(input_label_dir, label_filename)
        dest_label_path = os.path.join(output_label_dir, label_filename)

        if os.path.exists(source_label_path):
            shutil.copy2(source_label_path, dest_label_path)


def main(input_image_root, input_label_root, output_image_root, output_label_root):
    """
    Main function to find all sequences and process them.
    """
    if not os.path.isdir(input_image_root):
        print(f"Error: Input image directory not found: {input_image_root}")
        return
    if not os.path.isdir(input_label_root):
        print(f"Error: Input label directory not found: {input_label_root}")
        return

    if not os.path.exists(output_image_root):
        os.makedirs(output_image_root)
        print(f"Created output image directory: {output_image_root}")
    if not os.path.exists(output_label_root):
        os.makedirs(output_label_root)
        print(f"Created output label directory: {output_label_root}")

    # Get all sequence directories (data01, data02, etc.) from the image root
    sequence_dirs = sorted([d for d in os.listdir(input_image_root) if os.path.isdir(os.path.join(input_image_root, d))])

    print(f"Found {len(sequence_dirs)} sequences to process.")

    for seq_dir in sequence_dirs:
        input_image_path = os.path.join(input_image_root, seq_dir)
        input_label_path = os.path.join(input_label_root, seq_dir)
        output_image_path = os.path.join(output_image_root, seq_dir)
        output_label_path = os.path.join(output_label_root, seq_dir)
        process_sequence(input_image_path, input_label_path, output_image_path, output_label_path)

    print("\nAll sequences processed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform frame differencing on image sequences and copy corresponding labels.")
    parser.add_argument('--input-images', type=str, default='tiaozhanbei_datasets_1/images',
                        help='Root directory of the input image sequences.')
    parser.add_argument('--input-labels', type=str, default='tiaozhanbei_datasets_1/labels',
                        help='Root directory of the input label sequences.')
    parser.add_argument('--output-images', type=str, default='tiaozhanbei_datasets_1/images_diff',
                        help='Root directory for the output processed images.')
    parser.add_argument('--output-labels', type=str, default='tiaozhanbei_datasets_1/labels_diff',
                        help='Root directory for the output processed labels.')

    args = parser.parse_args()

    main(args.input_images, args.input_labels, args.output_images, args.output_labels) 