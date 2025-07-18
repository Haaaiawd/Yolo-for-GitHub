import argparse
import os
from pathlib import Path


def fix_path_file(input_path, output_path):
    """
    Reads a file containing a list of paths, converts them to project-relative paths,
    and writes them to a new file.

    It assumes the relevant part of the path starts with 'tiaozhanbei_datasets_1'
    and should be prefixed with 'datasets/' to be correct.
    """
    print(f"Processing {input_path} -> {output_path}...")
    count = 0
    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            # Use pathlib to handle OS-specific paths and convert to generic format
            p = Path(line)
            try:
                # Find the index of the 'tiaozhanbei_datasets_1' part
                idx = p.parts.index('tiaozhanbei_datasets_1')
                # Reconstruct the path relative to the project root ('Yolo/')
                # e.g., parts = ('G:', '/', 'Yolo', 'tiaozhanbei_datasets_1', 'images_diff', ...)
                # We want 'datasets' + 'tiaozhanbei_datasets_1' + ...
                relative_parts = ['datasets'] + list(p.parts[idx:])
                relative_path = Path(*relative_parts)
                
                # Write the path with forward slashes for consistency
                f_out.write(relative_path.as_posix() + '\n')
                count += 1
            except ValueError:
                print(f"  - Warning: Could not find 'tiaozhanbei_datasets_1' in line: {line}. Skipping.")
    print(f"Processed {count} paths.")


def main():
    parser = argparse.ArgumentParser(description="Fix hardcoded absolute paths in dataset list files.")
    parser.add_argument('--train-in', type=str, default='configs/train.txt', help='Input train file.')
    parser.add_argument('--val-in', type=str, default='configs/val.txt', help='Input val file.')
    parser.add_argument('--train-out', type=str, default='configs/train_fixed.txt', help='Output train file.')
    parser.add_argument('--val-out', type=str, default='configs/val_fixed.txt', help='Output val file.')
    args = parser.parse_args()

    fix_path_file(args.train_in, args.train_out)
    fix_path_file(args.val_in, args.val_out)
    print("\nPath files fixed successfully!")


if __name__ == '__main__':
    main() 