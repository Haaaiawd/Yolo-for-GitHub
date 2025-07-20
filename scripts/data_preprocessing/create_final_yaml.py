#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终配置文件生成脚本 - 云端版本
用于在云端环境中生成训练配置文件
"""

import os
import sys
import yaml

def main():
    """
    生成最终的YAML配置文件
    """
    print("📝 生成最终配置文件...")
    
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    dataset_dir = os.path.join(project_root, 'datasets', 'balanced_tiaozhanbei_split')
    config_dir = os.path.join(project_root, 'Yolo-for-GitHub', 'configs')
    
    print(f"数据集目录: {dataset_dir}")
    print(f"配置目录: {config_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"❌ 错误: 数据集目录不存在: {dataset_dir}")
        sys.exit(1)
    
    # 检查数据集统计
    stats = {}
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(dataset_dir, 'images', split)
        labels_dir = os.path.join(dataset_dir, 'labels', split)
        flow_dir = os.path.join(dataset_dir, 'flow', split)
        
        if os.path.exists(images_dir):
            image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.bmp'))])
            stats[split] = {'images': image_count}
            print(f"  {split}: {image_count} 图像")
        
        if os.path.exists(flow_dir):
            flow_count = len([f for f in os.listdir(flow_dir) if f.endswith('.npy')])
            stats[split]['flow'] = flow_count
            print(f"  {split}: {flow_count} 光流文件")
    
    # 生成云端配置文件
    config = {
        'path': '../datasets/balanced_tiaozhanbei_split',
        'train': '../datasets/balanced_tiaozhanbei_split/images/train',
        'val': '../datasets/balanced_tiaozhanbei_split/images/val',
        'test': '../datasets/balanced_tiaozhanbei_split/images/test',
        'nc': 6,
        'names': {
            0: 'drone',
            1: 'bus', 
            2: 'ship',
            3: 'car',
            4: 'cyclist',
            5: 'pedestrian'
        }
    }
    
    # 保存配置文件
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'tiaozhanbei_cloud.yaml')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 配置文件已生成: {config_path}")
    
    # 显示配置内容
    print("\n📋 配置文件内容:")
    with open(config_path, 'r', encoding='utf-8') as f:
        print(f.read())

if __name__ == '__main__':
    main()
