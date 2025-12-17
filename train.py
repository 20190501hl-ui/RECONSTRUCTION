#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多阶段重构攻击完整训练流程
一键运行完整的攻击训练pipeline
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
        # 首先尝试默认加载
        return torch.load(file_path)
    except (RuntimeError, pickle.PicklingError, Exception) as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["weights_only", "pil.image", "weightspickler", "unsupported global"]):
            # 如果是PyTorch 2.6+的安全加载问题，使用weights_only=False
            print(f"使用兼容模式加载 {file_path} (PyTorch 2.6+)")
            return torch.load(file_path, weights_only=False)
        else:
            raise e

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_banner():
    """打印项目横幅"""
    print("=" * 70)
    print("多阶段重构攻击训练系统 (特征提取+分类器)")
    print("   Multi-stage Attack Training (Feature Extraction + Classifier)")
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
        "单阶段图像": "result/single_stage_images",
        "多阶段图像": "result/multi_stage_images",
        "结果目录": "result",
        "模型目录": "model"
    }
    
    missing = []
    for name, path in required_paths.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                if name.endswith("图像"):
                    count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
                    print(f"OK {name}: {path} ({count} 张图片)")
                else:
                    print(f"OK {name}: {path}")
            else:
                print(f"OK {name}: {path}")
        else:
            print(f"MISS {name}: {path} (缺失)")
            missing.append((name, path))
    
    if missing:
        print(f"\nERR 缺失必要文件/目录:")
        for name, path in missing:
            print(f"   - {name}: {path}")
        
        # 检查是否缺少图像
        image_missing = any("图像" in name for name, path in missing)
        if image_missing:
            print("\nTIP 解决方案:")
            print("请先运行图像生成: python generate_images.py")
        
        return False
    
    print(f"\nOK 必要文件检查通过")
    return True

def run_pipeline():
    """运行训练pipeline（跳过图像生成）"""
    
    steps = [
        {
            "name": "特征提取", 
            "desc": "提取1维和6维特征向量",
            "module": "extract_features",
            "time_est": "5-15分钟"
        },
        {
            "name": "分类器训练",
            "desc": "训练攻击分类器并评估性能",
            "module": "classifier", 
            "time_est": "1-5分钟"
        }
    ]
    
    print(f"\n开始执行训练流程 (预计总时间: 10-20分钟)")
    print("注意: 跳过图像生成步骤，直接从特征提取开始")
    print("=" * 50)
    
    for i, step in enumerate(steps, 1):
        print(f"\n步骤 {i}/2: {step['name']}")
        print(f"描述: {step['desc']}")
        print(f"预计耗时: {step['time_est']}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # 动态导入并运行模块
            module = __import__(step['module'])
            print(f"开始执行 {step['module']}.py...")
            
            # 运行主函数
            module.main()
            
            elapsed_time = time.time() - start_time
            print(f"OK {step['name']} 完成 (耗时: {elapsed_time:.1f}秒)")
            
        except Exception as e:
            print(f"ERR {step['name']} 失败: {e}")
            print(f"错误详情: {str(e)}")
            return False
    
    return True

def show_results():
    """显示训练结果"""
    print(f"\n训练结果总结")
    print("=" * 50)
    
    # 检查生成的文件
    result_files = {
        "单阶段特征": "result/single_stage_features.pt",
        "多阶段特征": "result/multi_stage_features.pt",
        "对比结果": "result/models/comparison_results.pt"
    }
    
    print("\n生成文件:")
    for name, path in result_files.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                count = len(os.listdir(path))
                print(f"OK {name}: {path} ({count} 文件)")
            else:
                print(f"OK {name}: {path}")
        else:
            print(f"MISS {name}: {path} (未找到)")
    
    # 显示性能结果
    results_path = "result/models/comparison_results.pt"
    if os.path.exists(results_path):
        try:
            results = safe_torch_load(results_path)
            single_res = results.get('single', {})
            multi_res = results.get('multi', {})
            
            print(f"\n性能对比:")
            print(f"{'方法':<15} {'准确率':<10} {'AUC-ROC':<10}")
            print("-" * 35)
            
            single_acc = single_res.get('accuracy', 0)
            single_auc = single_res.get('auc_roc', 0)
            multi_acc = multi_res.get('accuracy', 0)
            multi_auc = multi_res.get('auc_roc', 0)
            
            print(f"{'单阶段方法':<15} {single_acc:<10.4f} {single_auc:<10.4f}")
            print(f"{'多阶段方法':<15} {multi_acc:<10.4f} {multi_auc:<10.4f}")
            
            if multi_acc > single_acc:
                improvement = multi_acc - single_acc
                print(f"\nGOOD 性能提升: +{improvement:.4f} ({improvement/single_acc*100:.1f}%)")
            
        except Exception as e:
            print(f"WARN 无法加载结果文件: {e}")
    
    print(f"\n[SUCCESS] 特征提取和分类器训练完成!")
    print(f"接下来可运行: python simple_test.py 查看详细结果")

def main():
    """主函数"""
    print_banner()
    
    # 检查前提条件
    if not check_prerequisites():
        print(f"\nERR 前提条件检查失败，请解决上述问题后重试")
        sys.exit(1)
    
    print(f"\nOK 前提条件检查通过")
    
    # 询问用户确认
    print(f"\n是否开始训练? (这将花费一些时间)")
    response = input("输入 'y' 或 'yes' 确认，其他键取消: ").lower().strip()
    
    if response not in ['y', 'yes']:
        print("STOP 用户取消训练")
        return
    
    # 创建必要目录
    os.makedirs("result", exist_ok=True)
    os.makedirs("result/models", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    os.makedirs("model/classifiers", exist_ok=True)
    os.makedirs("model/embeddings", exist_ok=True)
    os.makedirs("model/configs", exist_ok=True)
    
    # 运行训练流程
    success = run_pipeline()
    
    if success:
        show_results()
        print(f"\nSUCC 特征提取和分类器训练完成!")
    else:
        print(f"\nERR 训练流程失败，请检查错误信息")
    
    print("=" * 70)

if __name__ == "__main__":
    main()