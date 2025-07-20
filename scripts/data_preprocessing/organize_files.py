#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件整理脚本 - 云端版本
用于在云端环境中整理数据集文件结构
"""

import os
import sys
import shutil
from pathlib import Path

def main():
    """
    文件整理主函数
    检查数据集结构，如果需要则进行整理
    """
    print("📁 检查数据集文件结构...")
    
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    dataset_dir = os.path.join(project_root, 'datasets', 'balanced_tiaozhanbei_split')
    
    print(f"数据集目录: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"❌ 错误: 数据集目录不存在: {dataset_dir}")
        print("请确保数据集已上传到正确位置")
        sys.exit(1)
    
    # 检查必要的目录结构
    required_dirs = [
        'images/train',
        'images/val', 
        'images/test',
        'labels/train',
        'labels/val',
        'labels/test'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_dir, dir_path)
        if not os.path.exists(full_path):
            missing_dirs.append(dir_path)
        else:
            file_count = len([f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))])
            print(f"  ✅ {dir_path}: {file_count} 文件")
    
    if missing_dirs:
        print(f"❌ 缺失目录: {missing_dirs}")
        sys.exit(1)
    
    # 创建flow目录结构
    flow_dirs = ['flow/train', 'flow/val', 'flow/test']
    for dir_path in flow_dirs:
        full_path = os.path.join(dataset_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"  📁 创建/确认目录: {dir_path}")
    
    print("✅ 文件结构检查完成")

if __name__ == '__main__':
    main()
