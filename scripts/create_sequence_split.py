import os
import random
import shutil
import argparse
from pathlib import Path
import yaml

def get_sequence_ids(image_root: Path):
    """从图像目录中获取所有视频序列ID（例如 data01, data02）。"""
    if not image_root.is_dir():
        print(f"错误: 源图像目录不存在: {image_root}")
        return []
    return sorted([p.name for p in image_root.iterdir() if p.is_dir() and p.name.startswith('data')])

def create_dirs(base_path: Path):
    """在目标路径下创建标准的YOLOv8 images/labels 和 train/val 目录结构。"""
    for split in ['train', 'val']:
        for data_type in ['images', 'labels']:
            (base_path / data_type / split).mkdir(parents=True, exist_ok=True)

def split_sequences(sequence_ids, train_ratio=0.8, seed=42):
    """按序列ID的列表将数据集分割为训练集和验证集。"""
    random.seed(seed)
    shuffled_ids = sequence_ids.copy()
    random.shuffle(shuffled_ids)
    split_idx = int(len(shuffled_ids) * train_ratio)
    train_ids = shuffled_ids[:split_idx]
    val_ids = shuffled_ids[split_idx:]
    return train_ids, val_ids

def copy_sequence_data(sequence_ids, split_name, source_images_dir: Path, source_labels_dir: Path, dest_root: Path):
    """将一个完整序列的所有数据（图像和标签）复制到目标目录。"""
    for seq_id in sequence_ids:
        # 复制图像序列
        src_img_seq_dir = source_images_dir / seq_id
        dest_img_seq_dir = dest_root / 'images' / split_name / seq_id
        if src_img_seq_dir.is_dir():
            shutil.copytree(src_img_seq_dir, dest_img_seq_dir)

        # 复制标签序列
        src_lbl_seq_dir = source_labels_dir / seq_id
        dest_lbl_seq_dir = dest_root / 'labels' / split_name / seq_id
        if src_lbl_seq_dir.is_dir():
            shutil.copytree(src_lbl_seq_dir, dest_lbl_seq_dir)
    print(f"成功复制 {len(sequence_ids)} 个序列到 {split_name} 集。")

def create_yaml_config(dest_root: Path, class_names: list):
    """在目标根目录创建YOLOv8数据集配置文件(data.yaml)。"""
    # 使用相对路径以保证可移植性
    config_data = {
        'path': '.',  
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    config_path = dest_root / 'tiaozhanbei_seq_data.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return config_path

def main(args):
    source_root = Path(args.source_dir)
    dest_root = Path(args.dest_dir)
    
    source_images = source_root / "images_diff"
    source_labels = source_root / "labels_diff"

    print(f"开始处理数据集...")
    print(f"源数据目录: {source_root}")
    print(f"目标目录: {dest_root}")

    sequence_ids = get_sequence_ids(source_images)
    if not sequence_ids:
        print("在源目录中未发现序列文件夹 (如 'data01', 'data02', ...)。请检查路径。")
        return
    print(f"发现 {len(sequence_ids)} 个视频序列。")

    train_ids, val_ids = split_sequences(sequence_ids, args.train_ratio)
    print(f"分割完成 -> 训练序列: {len(train_ids)} 个, 验证序列: {len(val_ids)} 个。")

    print("正在创建目标目录结构...")
    create_dirs(dest_root)

    print("正在复制训练数据...")
    copy_sequence_data(train_ids, 'train', source_images, source_labels, dest_root)
    
    print("正在复制验证数据...")
    copy_sequence_data(val_ids, 'val', source_images, source_labels, dest_root)

    class_names = ['drone', 'car', 'ship', 'bus', 'pedestrian', 'cyclist']
    print("正在创建数据集配置文件...")
    config_path = create_yaml_config(dest_root, class_names)

    print("-" * 30)
    print(f"✅ 数据集重建完成，符合CQ-09高帧频要求。")
    print(f"配置文件已创建于: {config_path}")
    print("现在您可以在训练时使用此配置文件。")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按视频序列分割YOLO数据集，以符合高帧频任务要求。")
    parser.add_argument('--source-dir', type=str, required=True, help="包含 'images_diff' 和 'labels_diff' 的源数据集根目录的路径。")
    parser.add_argument('--dest-dir', type=str, required=True, help="用于保存按序列分割后的新数据集的目标目录路径。")
    parser.add_argument('--train-ratio', type=float, default=0.8, help="训练集所占的序列比例。")
    
    args = parser.parse_args()
    main(args) 