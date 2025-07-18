import os
import random
import argparse
import shutil
from tqdm import tqdm

def move_files(files, dest_image_dir, dest_label_dir, source_label_root):
    """
    Moves a list of image files and their corresponding label files to the destination directories.
    """
    for img_path in tqdm(files, desc=f"Moving to {os.path.basename(dest_image_dir)}"):
        try:
            # Move the image file
            shutil.move(img_path, dest_image_dir)

            # Determine the corresponding label file path
            # img_path is like: .../images_diff/dataXX/filename.bmp
            # We need to find: .../labels_diff/dataXX/filename.txt
            base_filename = os.path.basename(img_path)
            sequence_dir = os.path.basename(os.path.dirname(img_path)) # e.g., dataXX
            label_filename = os.path.splitext(base_filename)[0] + '.txt'
            
            source_label_path = os.path.join(source_label_root, sequence_dir, label_filename)

            # Move the label file if it exists
            if os.path.exists(source_label_path):
                shutil.move(source_label_path, dest_label_dir)
        except Exception as e:
            print(f"Warning: Could not move file {img_path} or its label. Reason: {e}")


def create_data_split_and_move(source_image_root, source_label_root, dest_root, train_split=0.8):
    """
    Scans source directories, splits files into train/val sets, and moves them
    to a new structured directory.
    """
    # Define destination paths
    dest_train_img = os.path.join(dest_root, 'images', 'train')
    dest_val_img = os.path.join(dest_root, 'images', 'val')
    dest_train_lbl = os.path.join(dest_root, 'labels', 'train')
    dest_val_lbl = os.path.join(dest_root, 'labels', 'val')

    # Ensure all destination directories exist
    os.makedirs(dest_train_img, exist_ok=True)
    os.makedirs(dest_val_img, exist_ok=True)
    os.makedirs(dest_train_lbl, exist_ok=True)
    os.makedirs(dest_val_lbl, exist_ok=True)

    # --- Start of new logic to handle existing files ---
    # To make this script runnable multiple times, check if the source dir is empty.
    # If not, it means we should probably not run it again, or we need to clean up first.
    if not os.listdir(source_image_root):
        print("Source image directory is empty. Nothing to move. Exiting.")
        return
        
    all_image_paths = []
    for root, _, files in os.walk(source_image_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.bmp')):
                all_image_paths.append(os.path.join(root, file))
    
    if not all_image_paths:
        print("Error: No images found in the source directory.")
        return

    random.shuffle(all_image_paths)

    split_index = int(len(all_image_paths) * train_split)
    train_files = all_image_paths[:split_index]
    val_files = all_image_paths[split_index:]

    print("Starting to move training files...")
    move_files(train_files, dest_train_img, dest_train_lbl, source_label_root)

    print("\nStarting to move validation files...")
    move_files(val_files, dest_val_img, dest_val_lbl, source_label_root)

    print("\n-----------------------------------------")
    print("Data restructuring completed successfully!")
    print(f"Total files moved: {len(all_image_paths)}")
    print(f"Training set size: {len(train_files)}")
    print(f"Validation set size: {len(val_files)}")
    print(f"Data is now located in: {dest_root}")
    print("-----------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Restructure dataset into YOLOv8 format.")
    parser.add_argument('--source_image_root', type=str, default='tiaozhanbei_datasets_1/images_diff',
                        help='Root directory of the source processed images.')
    parser.add_argument('--source_label_root', type=str, default='tiaozhanbei_datasets_1/labels_diff',
                        help='Root directory of the source processed labels.')
    parser.add_argument('--dest_root', type=str, default='tiaozhanbei_datasets_final',
                        help='Destination root directory for the structured dataset.')
    
    args = parser.parse_args()

    create_data_split_and_move(args.source_image_root, args.source_label_root, args.dest_root) 