# -*- coding: utf-8 -*-
"""
YOLOv11 训练脚本 - 红外小目标检测 (已适配光流输入)
基于轻量化国产大模型的高帧频弱小目标检测识别技术研究

@Project: CQ-09 挑战杯
@Dataset: tiaozhanbei_datasets_final 
@Classes: 6 (drone, car, ship, bus, pedestrian, cyclist)
"""
import argparse
import warnings
import torch
import numpy as np
import cv2
import os
import re
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose

warnings.filterwarnings('ignore')

# =================================================================================
# 核心修改：自定义数据加载器以支持5通道（RGB+光流）输入
# =================================================================================
class YOLOv11OpticalFlowDataset(YOLODataset):
    """
    一个自定义的YOLO数据集类，用于加载RGB图像和对应的光流数据，
    并将它们合并为5通道输入。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_input(self, idx):
        """获取并处理单个样本的输入数据，包括图像和光流。"""
        # 首先，使用父类的方法获取原始的RGB图像
        # 直接获取图像，不使用get_image_and_label避免复杂的变换
        img_path = self.im_files[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 获取当前图像的原始路径
        img_path = self.im_files[idx]
        
        flow = None
        try:
            # 健壮的路径推断逻辑
            # e.g., from '.../datasets/balanced_tiaozhanbei_split/images/train/data01/0002.jpg'
            # to '.../datasets/balanced_tiaozhanbei_split/flow/train/data01/flow_0001_to_0002.npy'
            path_parts = img_path.replace(os.sep, '/').split('/')
            images_idx = path_parts.index('images')
            
            # 构建基础的光流目录
            flow_dir_parts = path_parts[:images_idx] + ['flow'] + path_parts[images_idx + 1:-1]
            flow_dir = '/'.join(flow_dir_parts)

            # 解析帧文件名
            # 使用正则表达式匹配文件名中的序列号，例如 '..._001500.jpg' -> '..._', '001500'
            filename = path_parts[-1]
            match = re.match(r'(.+?_)?(\d+)\.(jpg|png|bmp)', filename, re.IGNORECASE)
            if not match:
                raise FileNotFoundError("无法从文件名解析帧号")

            prefix = match.group(1) or ''
            current_frame_num = int(match.group(2))
            frame_num_len = len(match.group(2))

            if current_frame_num > 0:
                prev_frame_num = current_frame_num - 1
                
                # 保持帧号的零填充格式
                prev_frame_str = str(prev_frame_num).zfill(frame_num_len)
                current_frame_str = str(current_frame_num).zfill(frame_num_len)

                # 构建光流文件名
                flow_filename = f"flow_{prefix}{prev_frame_str}_to_{prefix}{current_frame_str}.npy"
                flow_filepath = os.path.join(flow_dir, flow_filename)

                if os.path.exists(flow_filepath):
                    flow = np.load(flow_filepath)
                else:
                    # 如果推理出的文件不存在，也创建零光流
                    raise FileNotFoundError(f"光流文件不存在: {flow_filepath}")
            
        except (ValueError, IndexError, FileNotFoundError):
            # 如果是第一帧或任何路径解析失败，则创建一个全零的光流
            pass

        if flow is None:
            h, w, _ = img.shape
            flow = np.zeros((h, w, 2), dtype=np.float32)

        # 调整光流的尺寸以匹配（可能经过变换的）图像尺寸
        if flow.shape[:2] != img.shape[:2]:
            flow = cv2.resize(flow, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            
        # 核心：将RGB图像和光流数据在通道维度上合并
        # img (H, W, 3) + flow (H, W, 2) -> combined (H, W, 5)
        combined_input = np.concatenate((img, flow), axis=2)
        
        return combined_input

    def build_transforms(self, hyp=None):
        """
        构建数据增强/变换。
        注意：我们需要确保所有变换都能正确处理5通道数据。
        幸运的是，大部分ultralytics的几何变换（如翻转、缩放）是通道无关的。
        颜色空间的变换则需要注意，但默认情况下它们通常只应用于前3个通道。
        为简单起见，我们这里直接使用父类的方法，对于标准增强是安全的。
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            # 创建一个能处理5通道的Compose对象
            return Compose(self.get_transforms(hyp))
        else:
            # 对于验证集，通常只有Resize和ToTensor
            return self.get_transforms(hyp)

    def __getitem__(self, idx):
        """重写__getitem__方法，返回5通道数据和标签。"""
        # 获取标签等信息
        label = self.labels[idx].copy()
        
        # 获取5通道输入
        input_data = self.get_input(idx)
        
        # 获取变换
        transform = self.build_transforms(self.hyp)
        
        # 应用变换
        if transform:
            input_data, label = transform(input_data, label)

        # Transpose a ndarray from (H, W, C) to (C, H, W)
        return torch.from_numpy(input_data.transpose(2, 0, 1)), label


# =================================================================================
# 主训练逻辑
# =================================================================================
def main():
    parser = argparse.ArgumentParser(description="YOLOv11 训练脚本 (已适配光流输入)")
    # 重要：修改帮助文本，提示使用5通道模型
    parser.add_argument('--model', type=str, required=True, help='模型配置文件的路径 (e.g., configs/yolov11-5ch.yaml)')
    parser.add_argument('--data', type=str, required=True, help='数据集配置文件的路径 (e.g., configs/tiaozhanbei_final.yaml)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批处理大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--project', type=str, default='runs/train', help='项目保存目录')
    parser.add_argument('--name', type=str, default='exp_5channel', help='实验名称')
    parser.add_argument('--device', type=str, default='', help='运行设备，如 "cpu", "0", "0,1"')
    parser.add_argument('--save_period', type=int, default=-1, help='每隔N个epoch保存一次检查点, -1为不启用')
    # 核心修改：添加 --weights 参数用于迁移学习
    parser.add_argument('--weights', type=str, default='', help='预训练权重文件的路径 (e.g., path/to/yolo11x.pt)')
    # 对于5通道模型，通常需要从yaml构建，而不是从pt恢复，但保留resume选项以备不时之需
    parser.add_argument('--resume', type=str, default='', help='从指定的.pt文件恢复训练, e.g., path/to/last.pt')
    
    args = parser.parse_args()

    # 初始化YOLO模型
    print(f"正在加载模型: {args.model}")

    # 检查是否是预训练模型文件
    if args.model.endswith('.pt'):
        print("📦 基于预训练模型进行5通道扩展...")
        model = YOLO(args.model)

        # 修改第一层卷积以支持5通道输入
        def modify_first_conv_for_5_channels(model):
            """修改模型的第一层卷积以支持5通道输入"""
            import torch.nn as nn

            # 获取第一层卷积
            first_conv = model.model.model[0].conv

            # 创建新的5通道卷积层
            new_conv = nn.Conv2d(
                in_channels=5,  # 5通道输入
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )

            # 复制原有权重到新层
            with torch.no_grad():
                # 复制RGB通道的权重
                new_conv.weight[:, :3, :, :] = first_conv.weight
                # 为光流通道初始化权重（复制绿色通道的权重）
                new_conv.weight[:, 3:5, :, :] = first_conv.weight[:, 1:2, :, :].repeat(1, 2, 1, 1)

                if first_conv.bias is not None:
                    new_conv.bias = first_conv.bias

            # 替换第一层
            model.model.model[0].conv = new_conv
            print("✅ 成功将第一层卷积从3通道扩展到5通道")

            return model

        # 应用5通道修改
        model = modify_first_conv_for_5_channels(model)

    else:
        print("📋 从YAML配置文件创建模型...")
        model = YOLO(args.model)


    # =============================================================================
    # 核心修改：将自定义的Dataset类注入到训练器中
    # =============================================================================

    # 检查是否使用5通道模型
    is_5channel = args.model.endswith('5ch.yaml') or '5ch' in args.model

    if is_5channel:
        print("🚀 使用5通道模型（RGB + 光流）进行训练...")

        # 为5通道模型设置自定义数据集
        from ultralytics.engine.trainer import BaseTrainer
        from ultralytics.models.yolo.detect import DetectionTrainer

        # 创建自定义训练器类
        class OpticalFlowTrainer(DetectionTrainer):
            def build_dataset(self, img_path, mode="train", batch=None):
                """构建自定义数据集，支持5通道输入"""
                dataset = YOLOv11OpticalFlowDataset(
                    img_path=img_path,
                    imgsz=self.args.imgsz,
                    batch_size=batch,
                    augment=mode == "train",
                    hyp=self.hyp,
                    rect=False,
                    cache=getattr(self.args, 'cache', False),
                    single_cls=getattr(self.args, 'single_cls', False),
                    stride=int(self.stride.max() if self.stride else 32),
                    pad=0.0 if mode == "train" else 0.5,
                    prefix=f"{mode}: ",
                    task=self.args.task,
                    classes=getattr(self.args, 'classes', None),
                    data=self.data,
                    fraction=getattr(self.args, 'fraction', 1.0) if mode == "train" else 1.0,
                )
                return dataset

        # 使用自定义训练器
        model.trainer = OpticalFlowTrainer

    else:
        print("📷 使用标准3通道模型进行训练...")

    # 开始训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        save_period=args.save_period,
        resume=bool(args.resume),
        verbose=True
    )

    print("✅ 训练完成！")
    return results

if __name__ == '__main__':
    main()
