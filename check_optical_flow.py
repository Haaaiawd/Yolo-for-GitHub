#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光流数据检查脚本
用于检查光流数据的完整性和质量
"""

import os
import numpy as np
from pathlib import Path

def check_flow_data():
    """检查光流数据的完整性"""
    print("🌊 检查光流数据状态...")
    print("=" * 50)
    
    # 获取数据集路径
    dataset_dir = Path("../datasets/balanced_tiaozhanbei_split")
    
    if not dataset_dir.exists():
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        return False
    
    total_flow_files = 0
    total_image_files = 0
    
    for split in ['train', 'val', 'test']:
        print(f"\n📁 检查 {split.upper()} 集...")
        
        images_dir = dataset_dir / 'images' / split
        flow_dir = dataset_dir / 'flow' / split
        
        if not images_dir.exists():
            print(f"   ❌ 图像目录不存在: {images_dir}")
            continue
            
        # 统计图像文件
        image_files = list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.png")) + list(images_dir.rglob("*.bmp"))
        image_count = len(image_files)
        total_image_files += image_count
        print(f"   📷 图像文件: {image_count}")
        
        if not flow_dir.exists():
            print(f"   ⚠️  光流目录不存在: {flow_dir}")
            print(f"   💡 需要计算光流数据")
            continue
            
        # 统计光流文件
        flow_files = list(flow_dir.rglob("*.npy"))
        flow_count = len(flow_files)
        total_flow_files += flow_count
        print(f"   🌊 光流文件: {flow_count}")
        
        # 检查光流文件质量（抽样检查）
        if flow_files:
            sample_flow = flow_files[0]
            try:
                flow_data = np.load(sample_flow)
                print(f"   ✅ 光流数据格式正确: {flow_data.shape}")
                
                # 检查光流数据的合理性
                flow_magnitude = np.sqrt(flow_data[:,:,0]**2 + flow_data[:,:,1]**2)
                max_magnitude = np.max(flow_magnitude)
                mean_magnitude = np.mean(flow_magnitude)
                print(f"   📊 光流统计 - 最大值: {max_magnitude:.2f}, 平均值: {mean_magnitude:.2f}")
                
            except Exception as e:
                print(f"   ❌ 光流文件损坏: {e}")
    
    print("\n" + "=" * 50)
    print("📊 总体统计:")
    print(f"   总图像文件: {total_image_files}")
    print(f"   总光流文件: {total_flow_files}")
    
    if total_flow_files == 0:
        print("   🚨 没有光流数据，需要运行光流计算")
        print("   💡 运行: python scripts/data_preprocessing/compute_optical_flow.py")
        return False
    elif total_flow_files < total_image_files * 0.8:  # 假设每个图像序列至少有80%的光流
        print("   ⚠️  光流数据不完整，建议重新计算")
        return False
    else:
        print("   ✅ 光流数据完整")
        return True

def estimate_flow_computation_time():
    """估算光流计算时间"""
    print("\n⏱️  光流计算时间估算...")
    
    dataset_dir = Path("../datasets/balanced_tiaozhanbei_split")
    
    if not dataset_dir.exists():
        print("   ❌ 无法估算：数据集不存在")
        return
    
    total_images = 0
    for split in ['train', 'val', 'test']:
        images_dir = dataset_dir / 'images' / split
        if images_dir.exists():
            image_count = len(list(images_dir.rglob("*.jpg")) + 
                            list(images_dir.rglob("*.png")) + 
                            list(images_dir.rglob("*.bmp")))
            total_images += image_count
    
    if total_images > 0:
        # 估算：每个图像对计算光流约需要0.1-0.5秒（取决于GPU性能）
        estimated_pairs = total_images * 0.8  # 假设80%的图像能形成光流对
        
        print(f"   📊 总图像数: {total_images}")
        print(f"   🔄 预计光流对数: {int(estimated_pairs)}")
        print(f"   ⏱️  预计计算时间:")
        print(f"      - T4 GPU: {estimated_pairs * 0.1 / 60:.1f} - {estimated_pairs * 0.3 / 60:.1f} 分钟")
        print(f"      - CPU: {estimated_pairs * 2 / 60:.1f} - {estimated_pairs * 5 / 60:.1f} 分钟")
    else:
        print("   ❌ 无法估算：没有找到图像文件")

def main():
    """主函数"""
    print("🔍 光流数据完整性检查")
    print("=" * 50)
    
    flow_ok = check_flow_data()
    estimate_flow_computation_time()
    
    print("\n" + "=" * 50)
    if flow_ok:
        print("🎉 光流数据检查通过！可以直接开始训练")
    else:
        print("⚠️  需要计算光流数据")
        print("💡 建议操作:")
        print("   1. 运行: bash start_training.sh (自动计算光流)")
        print("   2. 或单独运行: python scripts/data_preprocessing/compute_optical_flow.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
