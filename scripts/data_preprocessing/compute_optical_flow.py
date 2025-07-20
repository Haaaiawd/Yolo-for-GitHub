#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光流计算脚本 - 云端优化版本
用于在云端环境中高效计算光流数据
"""

import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import time

def compute_flow_for_sequence(sequence_dir, flow_output_dir):
    """为一个独立的图像序列目录计算光流。"""
    print(f"  处理序列: {os.path.basename(sequence_dir)}")
    
    image_files = sorted([f for f in os.listdir(sequence_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    
    if len(image_files) < 2:
        print(f"    警告: 序列 {sequence_dir} 图像数量不足 (<2), 跳过。")
        return 0

    os.makedirs(flow_output_dir, exist_ok=True)
    
    # 读取第一帧
    first_img_path = os.path.join(sequence_dir, image_files[0])
    prev_gray = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
    
    if prev_gray is None:
        print(f"    错误: 无法读取第一帧 {first_img_path}")
        return 0

    flow_count = 0
    
    # 计算光流
    for i in range(1, len(image_files)):
        current_img_path = os.path.join(sequence_dir, image_files[i])
        current_gray = cv2.imread(current_img_path, cv2.IMREAD_GRAYSCALE)
        
        if current_gray is None:
            print(f"    警告: 无法读取图像 {current_img_path}, 跳过。")
            continue

        # 使用Farneback算法计算光流
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray, None, 
            pyr_scale=0.5,      # 图像金字塔的缩放参数
            levels=3,           # 金字塔层数
            winsize=15,         # 窗口大小
            iterations=3,       # 迭代次数
            poly_n=5,          # 像素邻域大小
            poly_sigma=1.2,    # 高斯标准差
            flags=0
        )
        
        # 生成光流文件名
        base1 = os.path.splitext(image_files[i-1])[0]
        base2 = os.path.splitext(image_files[i])[0]
        flow_filename = f"flow_{base1}_to_{base2}.npy"
        
        # 保存光流（添加错误处理和空间检查）
        flow_path = os.path.join(flow_output_dir, flow_filename)

        try:
            # 检查磁盘空间（更合理的检查逻辑）
            import shutil
            free_space = shutil.disk_usage(flow_output_dir).free
            estimated_size = flow.nbytes

            # 更合理的空间检查：只需要1.2倍缓冲，而不是2倍
            required_space = estimated_size * 1.2  # 20%缓冲就足够了

            if free_space < required_space:
                print(f"    ⚠️  磁盘空间不足！")
                print(f"       可用空间: {free_space/1024/1024:.1f}MB")
                print(f"       需要空间: {estimated_size/1024/1024:.1f}MB")
                print(f"       建议空间: {required_space/1024/1024:.1f}MB (含20%缓冲)")
                return flow_count

            # 如果空间紧张但足够，给出警告
            if free_space < estimated_size * 1.5:
                print(f"    ⚠️  磁盘空间紧张但足够继续：{free_space/1024/1024:.1f}MB可用")

            # 保存光流文件
            np.save(flow_path, flow)

            # 验证文件是否正确保存
            if os.path.exists(flow_path) and os.path.getsize(flow_path) > 0:
                # 快速验证文件可读性
                test_flow = np.load(flow_path)
                if test_flow.shape != flow.shape:
                    print(f"    ❌ 光流文件损坏，删除: {flow_filename}")
                    os.remove(flow_path)
                    continue
            else:
                print(f"    ❌ 光流文件保存失败: {flow_filename}")
                continue

        except OSError as e:
            print(f"    ❌ 保存光流文件时出错: {e}")
            print(f"    💾 检查磁盘空间: df -h")
            if os.path.exists(flow_path):
                os.remove(flow_path)  # 删除损坏的文件
            return flow_count
        except Exception as e:
            print(f"    ❌ 未知错误: {e}")
            if os.path.exists(flow_path):
                os.remove(flow_path)
            continue
        
        prev_gray = current_gray
        flow_count += 1

    print(f"    完成: 生成了 {flow_count} 个光流文件")
    return flow_count

def main():
    """
    主函数，遍历train/val/test下的每个序列子目录，并计算光流。
    云端优化版本，支持进度显示和错误处理。
    """
    print("🌊 开始批量计算光流（云端优化版本）...")
    print("=" * 60)
    
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    split_dir = os.path.join(project_root, 'datasets', 'balanced_tiaozhanbei_split')
    
    print(f"项目根目录: {project_root}")
    print(f"数据集目录: {split_dir}")
    
    if not os.path.exists(split_dir):
        print(f"❌ 错误: 数据集目录不存在: {split_dir}")
        print("请确保数据集已上传到正确位置")
        sys.exit(1)

    total_flow_files = 0
    start_time = time.time()

    for split in ['train', 'val', 'test']:
        print(f"\n📁 处理 {split.upper()} 集...")
        
        image_split_dir = os.path.join(split_dir, 'images', split)
        flow_split_dir = os.path.join(split_dir, 'flow', split)

        if not os.path.exists(image_split_dir):
            print(f"  ⚠️  目录不存在: {image_split_dir}, 跳过。")
            continue
            
        # 获取所有序列子目录
        sequence_dirs = [d for d in os.listdir(image_split_dir) 
                        if os.path.isdir(os.path.join(image_split_dir, d))]
        
        if not sequence_dirs:
            print(f"  ⚠️  在 {split} 集中未找到序列目录")
            continue
            
        print(f"  找到 {len(sequence_dirs)} 个序列")
        
        # 处理每个序列
        for seq_name in tqdm(sequence_dirs, desc=f"  计算 {split} 集光流"):
            seq_input_dir = os.path.join(image_split_dir, seq_name)
            seq_output_dir = os.path.join(flow_split_dir, seq_name)
            
            flow_count = compute_flow_for_sequence(seq_input_dir, seq_output_dir)
            total_flow_files += flow_count

    # 计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("✅ 光流计算完成！")
    print(f"📊 统计信息:")
    print(f"   - 总光流文件数: {total_flow_files}")
    print(f"   - 总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.1f} 分钟)")
    if total_flow_files > 0:
        print(f"   - 平均速度: {elapsed_time/total_flow_files:.2f} 秒/文件")
    print("=" * 60)

if __name__ == '__main__':
    main()
