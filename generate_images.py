#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像生成专用脚本
单独运行图像生成步骤，生成单阶段和多阶段重构图像
"""

import os
import sys
import time
import pickle
import torch
from datetime import datetime

def safe_torch_load(file_path):
    """安全加载包含PIL.Image对象的torch文件"""
    try:
        return torch.load(file_path)
    except (RuntimeError, pickle.PicklingError, Exception) as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["weights_only", "pil.image", "weightspickler", "unsupported global"]):
            print(f"使用兼容模式加载 {file_path} (PyTorch 2.6+)")
            return torch.load(file_path, weights_only=False)
        else:
            raise e

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_banner():
    """打印项目横幅"""
    print("=" * 70)
    print("多阶段重构攻击图像生成")
    print("   Multi-stage Reconstruction Attack Image Generation")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

def check_prerequisites():
    """检查运行前提条件"""
    print("\n检查运行前提条件...")
    
    required_paths = {
        "数据集": "data/test_dataset.pt",
        "结果目录": "result"
    }
    
    optional_paths = {
        "LoRA模型": "output_lora"
    }
    
    missing = []
    for name, path in required_paths.items():
        if os.path.exists(path):
            print(f"OK {name}: {path}")
        else:
            print(f"MISS {name}: {path} (缺失)")
            missing.append((name, path))
    
    # 检查可选文件
    for name, path in optional_paths.items():
        if os.path.exists(path):
            print(f"OK {name}: {path}")
        else:
            print(f"WARN {name}: {path} (可选，将使用基础模型)")
    
    if missing:
        print(f"\nERR 缺失必要文件/目录:")
        for name, path in missing:
            print(f"   - {name}: {path}")
        
        if "data/test_dataset.pt" in [p for n, p in missing]:
            print("\nTIP 解决方案:")
            print("1. 将数据集复制到 data/ 目录")
            print("2. 或运行: python test/test_functionality.py 创建测试数据集")
        
        return False
    
    print(f"\nOK 必要文件检查通过")
    if not os.path.exists("output_lora"):
        print(f"TIP 提示: 将使用基础Stable Diffusion模型，无需LoRA权重")
    
    return True

def run_image_generation():
    """运行图像生成步骤"""
    
    print(f"\n开始执行图像生成 (预计耗时: 10-30分钟)")
    print("=" * 50)
    print(f"\n步骤: 图像生成")
    print(f"描述: 生成单阶段和多阶段重构图像")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        # 动态导入并运行模块
        import inference
        print(f"开始执行 inference.py...")
        
        # 运行主函数
        inference.main()
        
        elapsed_time = time.time() - start_time
        print(f"OK 图像生成完成 (耗时: {elapsed_time:.1f}秒)")
        
        return True
        
    except Exception as e:
        print(f"ERR 图像生成失败: {e}")
        print(f"错误详情: {str(e)}")
        return False

def show_generation_results():
    """显示生成结果"""
    print(f"\n图像生成结果总结")
    print("=" * 50)
    
    # 检查生成的文件
    result_files = {
        "单阶段图像": "result/single_stage_images",
        "多阶段图像": "result/multi_stage_images"
    }
    
    print("\n生成文件:")
    for name, path in result_files.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
                print(f"OK {name}: {path} ({count} 张图片)")
            else:
                print(f"OK {name}: {path}")
        else:
            print(f"MISS {name}: {path} (未找到)")
    
    print(f"\n[SUCCESS] 图像生成完成!")
    print(f"接下来请运行: python train.py 继续特征提取和分类器训练")

def main():
    """主函数"""
    print_banner()
    
    # 检查前提条件
    if not check_prerequisites():
        print(f"\nERR 前提条件检查失败，请解决上述问题后重试")
        sys.exit(1)
    
    print(f"\nOK 前提条件检查通过")
    
    # 询问用户确认
    print(f"\n是否开始图像生成? (这将花费较长时间)")
    response = input("输入 'y' 或 'yes' 确认，其他键取消: ").lower().strip()
    
    if response not in ['y', 'yes']:
        print("STOP 用户取消图像生成")
        return
    
    # 创建必要目录
    os.makedirs("result", exist_ok=True)
    os.makedirs("result/single_stage_images", exist_ok=True)
    os.makedirs("result/multi_stage_images", exist_ok=True)
    
    # 运行图像生成
    success = run_image_generation()
    
    if success:
        show_generation_results()
    else:
        print(f"\nERR 图像生成失败，请检查错误信息")
    
    print("=" * 70)

if __name__ == "__main__":
    main()